import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.engine.chunked_scheduler import ChunkedPrefillScheduler
from mini_vllm.engine.sequence import Sequence, SequenceStatus
from mini_vllm.models.tiny_transformer import TinyTransformerConfig
from mini_vllm.models.tiny_transformer_paged import TinyTransformerPaged
from mini_vllm.layers.paged_kv_cache import PagedKVCache
from mini_vllm.models.tokenizer import CharTokenizer
from continuous_batching_generate import run_continuous_batching
from chunked_prefill_generate import run_chunked_prefill


def make_tokenizer_and_config(max_seq_len: int = 256):
    corpus = "the quick brown fox jumps over the lazy dog runs sleeps plays "
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
# Sequence：新增的 max_tokens / pending_token_count
# ---------------------------------------------------------------------------

def test_next_forward_input_without_cap_is_unchanged():
    """向後相容：不傳 max_tokens 時，行為要跟 Stage 3 完全一樣。"""
    seq = Sequence(seq_id="s", prompt_ids=[1, 2, 3, 4, 5], max_new_tokens=5)
    ids, start_pos = seq.next_forward_input()
    assert ids.tolist() == [[1, 2, 3, 4, 5]]
    assert start_pos == 0


def test_next_forward_input_with_cap_truncates():
    seq = Sequence(seq_id="s", prompt_ids=[1, 2, 3, 4, 5], max_new_tokens=5)
    ids, start_pos = seq.next_forward_input(max_tokens=2)
    assert ids.tolist() == [[1, 2]]
    assert start_pos == 0


def test_pending_token_count_reflects_uncomputed_tokens():
    seq = Sequence(seq_id="s", prompt_ids=[1, 2, 3, 4, 5], max_new_tokens=5)
    assert seq.pending_token_count == 5

    seq.mark_computed(2)
    assert seq.pending_token_count == 3

    seq.mark_computed(3)
    assert seq.pending_token_count == 0

    seq.append_token(9)
    assert seq.pending_token_count == 1  # 剛生成的新 token 還沒反映進 cache


# ---------------------------------------------------------------------------
# ChunkedPrefillScheduler：decode 優先、chunk 大小上限
# ---------------------------------------------------------------------------

def test_decode_ready_sequences_are_always_prioritized():
    bm = BlockManager(num_blocks=100, block_size=4)
    scheduler = ChunkedPrefillScheduler(bm, token_budget_per_step=3, max_prefill_chunk=100)

    decode_seq = Sequence(seq_id="decode", prompt_ids=[1], max_new_tokens=5)
    decode_seq.mark_computed(1)
    decode_seq.append_token(9)  # pending_token_count == 1，decode_ready
    prefill_seq = Sequence(seq_id="prefill", prompt_ids=list(range(20)), max_new_tokens=5)

    scheduler.add_request(decode_seq)
    scheduler.add_request(prefill_seq)
    schedule = scheduler.step()

    schedule_map = {seq.seq_id: n for seq, n in schedule}
    assert schedule_map["decode"] == 1  # decode 一定拿到它需要的 1 個 token
    assert schedule_map["prefill"] == 2  # 剩下的 budget（3-1=2）才給 prefill


def test_prefill_chunk_is_capped_by_max_prefill_chunk():
    bm = BlockManager(num_blocks=100, block_size=4)
    scheduler = ChunkedPrefillScheduler(bm, token_budget_per_step=100, max_prefill_chunk=5)

    seq = Sequence(seq_id="long", prompt_ids=list(range(50)), max_new_tokens=1)
    scheduler.add_request(seq)

    schedule = scheduler.step()
    assert schedule == [(seq, 5)]  # 即使 budget 很充裕，單一序列這步也只能拿 5 個


def test_long_prefill_is_spread_across_multiple_steps():
    bm = BlockManager(num_blocks=100, block_size=4)
    scheduler = ChunkedPrefillScheduler(bm, token_budget_per_step=100, max_prefill_chunk=5)

    seq = Sequence(seq_id="long", prompt_ids=list(range(12)), max_new_tokens=1)
    scheduler.add_request(seq)

    chunks = []
    for _ in range(5):
        schedule = scheduler.step()
        if not schedule:
            break
        for s, n in schedule:
            s.mark_computed(n)
            chunks.append(n)

    assert chunks == [5, 5, 2]  # 12 個 token，每步最多 5 個，分 3 步做完
    assert seq.pending_token_count == 0


def test_budget_too_small_for_even_one_decode_token_yields_empty_schedule():
    bm = BlockManager(num_blocks=100, block_size=4)
    scheduler = ChunkedPrefillScheduler(bm, token_budget_per_step=0, max_prefill_chunk=10)

    seq = Sequence(seq_id="s", prompt_ids=[1, 2, 3], max_new_tokens=5)
    scheduler.add_request(seq)

    # 就算 block 記憶體夠、seq 可以被准入 running，budget=0 這一步
    # 也真的什麼都不會處理。
    schedule = scheduler.step()
    assert schedule == []
    assert seq.status == SequenceStatus.RUNNING  # 有被准入，只是這步分不到 token


# ---------------------------------------------------------------------------
# 正確性驗證：chunked prefill 不能改變任何生成結果
# ---------------------------------------------------------------------------

def test_chunked_prefill_matches_unchunked_continuous_batching():
    """
    最重要的一個測試：把 prefill 切成很小的 chunk、跟 decode 混合
    排程之後，每個序列最終生成的文字，必須跟 Stage 3（不分 chunk，
    一次把整段 prefill 做完）完全相同。這證明 chunked prefill
    純粹是排程時機的調整，不影響任何數值計算的結果。
    """
    tokenizer, config = make_tokenizer_and_config()
    model = TinyTransformerPaged(config)

    requests = [
        ("seq-long", "the quick brown fox jumps over the lazy dog runs ", 6, 0),
        ("seq-a", "sleeps ", 8, 0),
        ("seq-b", "plays ", 8, 1),
    ]

    # Stage 3 風格：不分 chunk
    bm_unchunked = BlockManager(num_blocks=100, block_size=4)
    cache_unchunked = PagedKVCache(config, num_blocks=100, block_size=4)
    results_unchunked = run_continuous_batching(
        model, bm_unchunked, cache_unchunked, tokenizer, requests, verbose=False
    )

    # Stage 5 風格：chunk 切得很小，逼所有 prefill 都要跨好幾個 step
    bm_chunked = BlockManager(num_blocks=100, block_size=4)
    cache_chunked = PagedKVCache(config, num_blocks=100, block_size=4)
    scheduler_chunked = ChunkedPrefillScheduler(
        bm_chunked, token_budget_per_step=6, max_prefill_chunk=3
    )
    results_chunked, _ = run_chunked_prefill(
        model, scheduler_chunked, bm_chunked, cache_chunked, tokenizer, requests, verbose=False
    )

    assert results_unchunked == results_chunked


def test_all_sequences_finish_with_tight_budget():
    tokenizer, config = make_tokenizer_and_config()
    model = TinyTransformerPaged(config)

    bm = BlockManager(num_blocks=100, block_size=4)
    cache = PagedKVCache(config, num_blocks=100, block_size=4)
    scheduler = ChunkedPrefillScheduler(bm, token_budget_per_step=4, max_prefill_chunk=2)

    requests = [
        ("seq-1", "the quick brown fox ", 5, 0),
        ("seq-2", "jumps over ", 5, 0),
    ]
    results, _ = run_chunked_prefill(
        model, scheduler, bm, cache, tokenizer, requests, verbose=False
    )

    assert set(results.keys()) == {"seq-1", "seq-2"}
    assert bm.num_free_blocks == 100  # 全部做完後，池子完全恢復
