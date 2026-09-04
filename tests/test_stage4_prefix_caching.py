import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.engine.prefix_cache_block_manager import PrefixCachingBlockManager
from mini_vllm.layers.paged_kv_cache import PagedKVCache
from mini_vllm.models.tiny_transformer import TinyTransformerConfig
from mini_vllm.models.tiny_transformer_paged import TinyTransformerPaged
from mini_vllm.models.tokenizer import CharTokenizer
from paged_attention_generate import paged_generate
from prefix_caching_generate import prefix_cache_generate


def make_tokenizer_and_config(max_seq_len: int = 128):
    corpus = (
        "the quick brown fox jumps over the lazy dog runs sleeps plays "
        "and swims near the river "
    )
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
# PrefixCachingBlockManager 本身的單元測試
# ---------------------------------------------------------------------------

def test_identical_prompts_share_blocks_except_the_last_one():
    bm = PrefixCachingBlockManager(num_blocks=20, block_size=4)
    prompt_ids = list(range(17))  # 17 個 token -> ceil(17/4) = 5 個 block

    table_1, cached_1 = bm.allocate_prefill("seq-1", prompt_ids)
    table_2, cached_2 = bm.allocate_prefill("seq-2", prompt_ids)  # 完全相同的內容

    assert cached_1 == 0  # 第一個序列是冷啟動，什麼都沒命中
    assert cached_2 == 16  # 第二個序列命中前 4 個完整 block（16 個 token）

    # 前 4 個 block 應該是同一批實體 block（真的被共用）
    assert table_1[:4] == table_2[:4]
    # 最後一個 block 一定是各自私有的，不會共用
    assert table_1[4] != table_2[4]

    stats = bm.prefix_cache_stats()
    assert all(ref == 2 for ref in stats["shared_blocks"].values())


def test_diverging_prompts_share_only_common_prefix_blocks():
    bm = PrefixCachingBlockManager(num_blocks=20, block_size=4)

    common = list(range(8))  # 完整 2 個 block
    prompt_a = common + [100, 101, 102, 103, 104]  # 後面接不同內容
    prompt_b = common + [200, 201, 202]

    table_a, cached_a = bm.allocate_prefill("seq-a", prompt_a)
    table_b, cached_b = bm.allocate_prefill("seq-b", prompt_b)

    assert cached_b == 8  # 只有前面共同的 2 個完整 block 命中
    assert table_a[:2] == table_b[:2]
    assert table_a[2] != table_b[2]


def test_last_block_is_never_registered_even_if_content_repeats_later():
    """
    驗證『最後一個 block 永遠私有、不註冊進快取』這條規則真的有落實：
    即使兩個獨立的請求，其中一個請求的『非最後一個』完整 block，
    跟另一個請求『恰好也是完整、但屬於它自己 prompt 最後一個』的
    block 內容相同，也不應該被共用——因為第一個請求那個位置的
    block 是有註冊的，但因為 hash 是鏈式的（取決於它前面所有
    block），只要前綴不同就不會誤判成同一份內容。這裡改用更直接
    的方式驗證：同一個 prompt 分兩次呼叫，確認它自己的最後一個
    block 從未出現在 hash_to_block 索引裡。
    """
    bm = PrefixCachingBlockManager(num_blocks=20, block_size=4)
    prompt_ids = list(range(8))  # 剛好 2 個完整 block

    table, _ = bm.allocate_prefill("seq-1", prompt_ids)
    last_block_id = table[-1]

    assert last_block_id not in bm.block_content_hash


def test_freed_block_content_hash_persists_until_overwritten():
    """
    序列結束、block 被釋放（ref_count 歸零）之後，它的內容 hash
    索引應該還在——之後如果有新序列剛好需要一模一樣的前綴，
    應該能直接命中『復活』這個 block，不用重新配置或計算。
    """
    bm = PrefixCachingBlockManager(num_blocks=20, block_size=4)
    prompt_ids = list(range(8))

    table_1, _ = bm.allocate_prefill("seq-1", prompt_ids)
    shared_block = table_1[0]
    bm.free("seq-1")

    assert shared_block in bm.free_block_ids  # 回到 free pool
    assert shared_block in bm.block_content_hash  # 但 hash 索引還在

    table_2, cached_2 = bm.allocate_prefill("seq-2", prompt_ids)
    assert cached_2 == 4  # 命中了第一個 block
    assert table_2[0] == shared_block  # 而且真的是「復活」了同一個實體 block
    assert shared_block not in bm.free_block_ids  # 被借走了，不再是 free


def test_overwriting_a_freed_block_invalidates_its_old_hash():
    """
    如果一個被釋放的 block 被拿去存全新的內容（cache miss 配置），
    它舊的內容 hash 索引必須失效，不然之後會有人查到這個 block，
    卻讀到跟 hash 對不上的內容。
    """
    bm = PrefixCachingBlockManager(num_blocks=2, block_size=4)  # 剛好夠一個 5-token prompt 用，逼它一定被重複使用

    prompt_a = [1, 2, 3, 4, 99]  # 第 1 個 block 內容 [1,2,3,4]
    table_a, _ = bm.allocate_prefill("seq-a", prompt_a)
    old_hash = bm.block_content_hash[table_a[0]]
    bm.free("seq-a")

    prompt_b = [9, 9, 9, 9, 88]  # 完全不同的內容，只有 1 個 block 可用，一定會蓋掉舊的
    bm.allocate_prefill("seq-b", prompt_b)

    assert old_hash not in bm.hash_to_block  # 舊的內容 hash 已經失效


# ---------------------------------------------------------------------------
# 正確性驗證：Prefix caching 不能改變生成結果
# ---------------------------------------------------------------------------

def test_prefix_cache_hit_generation_matches_stage2_baseline():
    """
    最重要的一個測試：即使命中 prefix cache、跳過了部分 forward
    計算，最終生成的文字必須跟 Stage 2（完全不用 prefix caching，
    每個序列從頭到尾都自己重新算一遍）的結果逐字元相同。
    """
    tokenizer, config = make_tokenizer_and_config()
    model = TinyTransformerPaged(config)

    system_prompt = "the quick brown fox "
    prompt_2 = system_prompt + "runs and plays "
    max_new = 10

    # 有 prefix caching：先跑一次 seq-1 讓 system prompt 的 block 被註冊，
    # 再跑 seq-2，讓它實際命中快取。
    bm_cached = PrefixCachingBlockManager(num_blocks=30, block_size=4)
    cache_cached = PagedKVCache(config, num_blocks=30, block_size=4)
    prefix_cache_generate(
        model, bm_cached, cache_cached, tokenizer, "seq-1", system_prompt + "jumps ", max_new
    )
    text_with_cache, num_cached_tokens, _ = prefix_cache_generate(
        model, bm_cached, cache_cached, tokenizer, "seq-2", prompt_2, max_new
    )
    assert num_cached_tokens > 0  # 確認真的有命中，不然這個測試沒有意義

    # 沒有 prefix caching（Stage 2 的做法）：獨立的 pool，從頭跑
    bm_baseline = BlockManager(num_blocks=30, block_size=4)
    cache_baseline = PagedKVCache(config, num_blocks=30, block_size=4)
    text_baseline = paged_generate(
        model, bm_baseline, cache_baseline, tokenizer, "seq-2-baseline", prompt_2, max_new
    )

    assert text_with_cache == text_baseline


def test_multiple_sequences_sharing_prefix_all_match_baseline():
    tokenizer, config = make_tokenizer_and_config()
    model = TinyTransformerPaged(config)

    system_prompt = "the quick brown fox "
    requests = [
        ("seq-1", system_prompt + "jumps over ", 8),
        ("seq-2", system_prompt + "runs and plays ", 8),
        ("seq-3", system_prompt + "sleeps near the river ", 8),
    ]

    bm_cached = PrefixCachingBlockManager(num_blocks=40, block_size=4)
    cache_cached = PagedKVCache(config, num_blocks=40, block_size=4)

    cached_results = {}
    for seq_id, prompt, max_new in requests:
        text, _, _ = prefix_cache_generate(
            model, bm_cached, cache_cached, tokenizer, seq_id, prompt, max_new
        )
        cached_results[seq_id] = text

    baseline_results = {}
    for seq_id, prompt, max_new in requests:
        bm_iso = BlockManager(num_blocks=40, block_size=4)
        cache_iso = PagedKVCache(config, num_blocks=40, block_size=4)
        baseline_results[seq_id] = paged_generate(
            model, bm_iso, cache_iso, tokenizer, seq_id, prompt, max_new
        )

    assert cached_results == baseline_results


def test_pool_fully_recovers_after_all_sequences_freed():
    tokenizer, config = make_tokenizer_and_config()
    model = TinyTransformerPaged(config)

    num_blocks = 30
    bm = PrefixCachingBlockManager(num_blocks=num_blocks, block_size=4)
    cache = PagedKVCache(config, num_blocks=num_blocks, block_size=4)

    system_prompt = "the quick brown fox "
    seq_ids = []
    for i, suffix in enumerate(["jumps ", "runs ", "sleeps "]):
        seq_id = f"seq-{i}"
        seq_ids.append(seq_id)
        prefix_cache_generate(
            model, bm, cache, tokenizer, seq_id, system_prompt + suffix, 5
        )

    for seq_id in seq_ids:
        bm.free(seq_id)

    assert bm.num_free_blocks == num_blocks
    assert bm.ref_counts == {}
