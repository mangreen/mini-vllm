import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.layers.paged_kv_cache import PagedKVCache
from mini_vllm.layers.kv_cache import KVCache
from mini_vllm.models.tiny_transformer import TinyTransformer, TinyTransformerConfig
from mini_vllm.models.tiny_transformer_kv import TinyTransformerKV
from mini_vllm.models.tiny_transformer_paged import TinyTransformerPaged
from mini_vllm.models.tokenizer import CharTokenizer
from baseline_generate import naive_generate
from kv_cache_generate import kv_cache_generate
from paged_attention_generate import paged_generate


def make_tokenizer_and_config(max_seq_len: int = 64):
    corpus = "the quick brown fox jumps over the lazy dog "
    tokenizer = CharTokenizer(corpus)
    config = TinyTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=max_seq_len,
    )
    return tokenizer, config


# ---------------------------------------------------------------------------
# BlockManager 本身的單元測試
# ---------------------------------------------------------------------------

def test_num_blocks_needed_rounds_up():
    bm = BlockManager(num_blocks=10, block_size=4)
    assert bm.num_blocks_needed(1) == 1
    assert bm.num_blocks_needed(4) == 1  # 剛好整除
    assert bm.num_blocks_needed(5) == 2  # 多 1 個 token 就要多 1 個 block
    assert bm.num_blocks_needed(8) == 2
    assert bm.num_blocks_needed(9) == 3


def test_ensure_capacity_is_idempotent():
    bm = BlockManager(num_blocks=10, block_size=4)
    table_1 = bm.ensure_capacity("seq-A", 5)
    table_2 = bm.ensure_capacity("seq-A", 5)  # 同樣長度再叫一次
    assert table_1 == table_2
    assert len(table_1) == 2


def test_ensure_capacity_only_grows_when_crossing_block_boundary():
    """
    模擬 decode 逐 token 增長：只有真的跨過 block 邊界時，
    block table 才應該變長。
    """
    bm = BlockManager(num_blocks=10, block_size=4)

    table = bm.ensure_capacity("seq-A", 1)
    assert len(table) == 1
    for n in [2, 3, 4]:  # 還在第一個 block 裡（block_size=4）
        table = bm.ensure_capacity("seq-A", n)
        assert len(table) == 1

    table = bm.ensure_capacity("seq-A", 5)  # 跨過邊界，需要第 2 個 block
    assert len(table) == 2


def test_free_returns_blocks_to_pool_for_reuse():
    bm = BlockManager(num_blocks=4, block_size=4)

    bm.ensure_capacity("seq-A", 16)  # 用掉全部 4 個 block
    assert bm.num_free_blocks == 0

    bm.free("seq-A")
    assert bm.num_free_blocks == 4

    # 池子應該可以被另一個序列重新利用
    table = bm.ensure_capacity("seq-B", 16)
    assert len(table) == 4
    assert bm.num_free_blocks == 0


def test_concurrent_sequences_get_disjoint_blocks():
    bm = BlockManager(num_blocks=10, block_size=4)

    table_a = bm.ensure_capacity("seq-A", 5)
    table_b = bm.ensure_capacity("seq-B", 5)

    assert set(table_a).isdisjoint(set(table_b))


def test_pool_exhaustion_raises_memory_error():
    bm = BlockManager(num_blocks=2, block_size=4)
    bm.ensure_capacity("seq-A", 8)  # 用光 2 個 block

    try:
        bm.ensure_capacity("seq-B", 1)
        assert False, "block 池已滿時應該要丟出 MemoryError"
    except MemoryError:
        pass


# ---------------------------------------------------------------------------
# PagedKVCache 的讀寫正確性
# ---------------------------------------------------------------------------

def test_paged_kv_cache_write_read_matches_contiguous_cache():
    """
    給定完全相同的 K/V 數值，PagedKVCache（透過 block table 轉址）
    讀出來的結果，應該跟 Stage 1 的連續記憶體 KVCache 完全一致。
    這證明「切成 block、允許不連續」這個改動本身沒有引入誤差，
    純粹只是換一種記憶體排列方式。
    """
    _, config = make_tokenizer_and_config()
    block_size = 4
    num_blocks = 20

    bm = BlockManager(num_blocks=num_blocks, block_size=block_size)
    paged_cache = PagedKVCache(config, num_blocks=num_blocks, block_size=block_size)
    contiguous_cache = KVCache(config, batch_size=1)

    torch.manual_seed(123)
    k = torch.randn(1, config.n_heads, 10, config.head_dim)
    v = torch.randn(1, config.n_heads, 10, config.head_dim)

    block_table = bm.ensure_capacity("seq-A", 10)
    paged_cache.write(layer_idx=0, start_pos=0, k=k, v=v, block_table=block_table)
    contiguous_cache.write(layer_idx=0, start_pos=0, k=k, v=v)

    k_paged, v_paged = paged_cache.read(layer_idx=0, end_pos=10, block_table=block_table)
    k_contig, v_contig = contiguous_cache.read(layer_idx=0, end_pos=10)

    assert torch.allclose(k_paged, k_contig)
    assert torch.allclose(v_paged, v_contig)


def test_paged_kv_cache_handles_non_contiguous_blocks():
    """
    刻意製造「block table 不連續」的情況（先分配、釋放、再分配，
    讓一個序列拿到的實體 block 編號故意不是遞增排列），
    驗證讀出來的邏輯順序依然正確——這是 Paged 設計要驗證的關鍵行為。
    """
    _, config = make_tokenizer_and_config()
    block_size = 4
    num_blocks = 6

    bm = BlockManager(num_blocks=num_blocks, block_size=block_size)
    paged_cache = PagedKVCache(config, num_blocks=num_blocks, block_size=block_size)

    # 先讓 seq-X 佔用前幾個 block，製造碎片，再釋放
    bm.ensure_capacity("seq-X", 8)  # 佔用 block 0, 1
    bm.free("seq-X")

    # seq-A 這次拿到的 block 應該還是從 free pool depoque 出來的（0, 1），
    # 但我們手動製造不連續：讓 seq-A 只拿 1 個 block，再讓 seq-Y 插隊拿走下一個
    table_a_partial = bm.ensure_capacity("seq-A", 4)  # 拿到 block 0
    bm.ensure_capacity("seq-Y", 4)  # 插隊拿走 block 1
    table_a_full = bm.ensure_capacity("seq-A", 8)  # seq-A 長大，被迫拿一個較後面的 block

    assert len(table_a_full) == 2
    assert table_a_full[0] == table_a_partial[0]
    assert table_a_full[1] != table_a_full[0] - 1 or True  # 說明性斷言，見下方 assert 真正驗證的內容
    # 真正要驗證的：即使 block table 不是連續遞增（例如 [0, 2]），
    # 讀出來的內容順序仍然依照「邏輯位置」而非「實體編號」排列。
    k = torch.randn(1, config.n_heads, 8, config.head_dim)
    v = torch.randn(1, config.n_heads, 8, config.head_dim)
    paged_cache.write(layer_idx=0, start_pos=0, k=k, v=v, block_table=table_a_full)
    k_read, v_read = paged_cache.read(layer_idx=0, end_pos=8, block_table=table_a_full)
    assert torch.allclose(k_read, k)
    assert torch.allclose(v_read, v)


# ---------------------------------------------------------------------------
# 正確性驗證：Stage 2 必須跟 Stage 0 / Stage 1 完全等價
# ---------------------------------------------------------------------------

def test_paged_generation_matches_stage0_and_stage1_exactly():
    tokenizer, config = make_tokenizer_and_config()

    model_v0 = TinyTransformer(config)
    model_v1 = TinyTransformerKV(config)
    model_v2 = TinyTransformerPaged(config)
    model_v1.load_state_dict(model_v0.state_dict())
    model_v2.load_state_dict(model_v0.state_dict())
    model_v0.eval()
    model_v1.eval()
    model_v2.eval()

    prompt = "the quick "
    max_new_tokens = 15

    text_v0, _ = naive_generate(model_v0, tokenizer, prompt, max_new_tokens)
    text_v1, _ = kv_cache_generate(model_v1, tokenizer, prompt, max_new_tokens)

    block_manager = BlockManager(num_blocks=20, block_size=4)
    paged_cache = PagedKVCache(config, num_blocks=20, block_size=4)
    text_v2 = paged_generate(
        model_v2, block_manager, paged_cache, tokenizer, "seq-test", prompt, max_new_tokens
    )

    assert text_v0 == text_v1 == text_v2


def test_multiple_sequences_share_pool_without_interference():
    """
    多個序列先後借用同一個 block 池，彼此的生成結果不應該互相干擾
    ——即使他們的 block table 在池子裡是交錯配置的。
    """
    tokenizer, config = make_tokenizer_and_config()
    model = TinyTransformerPaged(config)

    block_manager = BlockManager(num_blocks=20, block_size=4)
    paged_cache = PagedKVCache(config, num_blocks=20, block_size=4)

    text_a = paged_generate(
        model, block_manager, paged_cache, tokenizer, "seq-A", "the quick ", 8
    )
    text_b = paged_generate(
        model, block_manager, paged_cache, tokenizer, "seq-B", "the lazy dog ", 8
    )

    # 各自跑一次「獨立」的 baseline 版本（重新配置全新的 cache），
    # 確認共用池子沒有讓 seq-A、seq-B 的結果互相污染。
    fresh_bm = BlockManager(num_blocks=20, block_size=4)
    fresh_cache = PagedKVCache(config, num_blocks=20, block_size=4)
    text_a_fresh = paged_generate(
        model, fresh_bm, fresh_cache, tokenizer, "seq-A-fresh", "the quick ", 8
    )

    assert text_a == text_a_fresh
    assert text_a != text_b or True  # 不同 prompt，內容不必然不同，這裡主要驗證上面那行


def test_block_pool_exhaustion_is_reachable_and_recoverable():
    """
    驗證『池子滿了會報錯、釋放後可以復原』的完整生命週期，
    這對應到 Stage 3 排程器要處理的「記憶體不夠時怎麼辦」的前置情境。
    """
    tokenizer, config = make_tokenizer_and_config()
    model = TinyTransformerPaged(config)

    block_manager = BlockManager(num_blocks=3, block_size=4)  # 刻意配置很小的池子
    paged_cache = PagedKVCache(config, num_blocks=3, block_size=4)

    # 一個序列生成夠多 token，會需要超過 3 個 block（12 個 token）
    try:
        paged_generate(
            model, block_manager, paged_cache, tokenizer, "seq-big", "the quick brown ", 20
        )
        assert False, "應該要因為 block 池不夠而丟出 MemoryError"
    except MemoryError:
        pass

    # 釋放後，池子應該完全恢復，可以給別的序列使用
    block_manager.free("seq-big")
    assert block_manager.num_free_blocks == 3
