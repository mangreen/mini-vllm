import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.engine.scheduler import Scheduler
from mini_vllm.engine.sequence import Sequence, SequenceStatus
from mini_vllm.models.tiny_transformer import TinyTransformerConfig
from mini_vllm.models.tiny_transformer_paged import TinyTransformerPaged
from mini_vllm.layers.paged_kv_cache import PagedKVCache
from mini_vllm.models.tokenizer import CharTokenizer
from paged_attention_generate import paged_generate
from continuous_batching_generate import run_continuous_batching


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
# Sequence 狀態機
# ---------------------------------------------------------------------------

def test_sequence_next_forward_input_covers_prefill_and_decode_uniformly():
    seq = Sequence(seq_id="s", prompt_ids=[1, 2, 3], max_new_tokens=5)

    # 初次 prefill：num_computed_tokens=0，應該回傳整個 prompt
    ids, start_pos = seq.next_forward_input()
    assert ids.tolist() == [[1, 2, 3]]
    assert start_pos == 0

    seq.mark_computed(3)
    seq.append_token(9)  # 生成出第 1 個新 token

    # 正常 decode：只回傳「還沒反映進 cache」的那 1 個新 token
    ids, start_pos = seq.next_forward_input()
    assert ids.tolist() == [[9]]
    assert start_pos == 3


def test_sequence_finishes_after_max_new_tokens():
    seq = Sequence(seq_id="s", prompt_ids=[1, 2], max_new_tokens=2)
    assert not seq.is_finished
    seq.append_token(9)
    assert not seq.is_finished
    seq.append_token(9)
    assert seq.is_finished


def test_sequence_reset_for_preemption_keeps_output_but_clears_progress():
    seq = Sequence(seq_id="s", prompt_ids=[1, 2], max_new_tokens=5)
    seq.mark_computed(2)
    seq.append_token(9)
    seq.append_token(8)

    seq.reset_for_preemption()

    assert seq.num_computed_tokens == 0
    assert seq.status == SequenceStatus.WAITING
    assert seq.output_ids == [9, 8]  # 已生成的內容不會被丟掉
    assert seq.needs_forward  # 全部要重新算


# ---------------------------------------------------------------------------
# Scheduler：准入與 FCFS
# ---------------------------------------------------------------------------

def test_scheduler_admits_waiting_sequence_when_capacity_available():
    bm = BlockManager(num_blocks=10, block_size=4)
    scheduler = Scheduler(bm)

    seq = Sequence(seq_id="s", prompt_ids=list(range(5)), max_new_tokens=3)
    scheduler.add_request(seq)

    running = scheduler.step()
    assert running == [seq]
    assert seq.status == SequenceStatus.RUNNING


def test_scheduler_leaves_sequence_waiting_when_not_enough_capacity():
    bm = BlockManager(num_blocks=1, block_size=4)  # 只有 1 個 block
    scheduler = Scheduler(bm)

    seq = Sequence(seq_id="s", prompt_ids=list(range(8)), max_new_tokens=3)  # 需要 2 個 block
    scheduler.add_request(seq)

    running = scheduler.step()
    assert running == []
    assert seq.status == SequenceStatus.WAITING


def test_scheduler_fcfs_does_not_let_smaller_request_skip_the_queue():
    """
    隊伍最前面的大請求塞不下時，即使後面排著一個明明塞得下的小請求，
    也不應該被插隊——這是 FCFS 政策要驗證的行為。
    """
    bm = BlockManager(num_blocks=2, block_size=4)  # 只有 2 個 block
    scheduler = Scheduler(bm)

    big = Sequence(seq_id="big", prompt_ids=list(range(12)), max_new_tokens=1)  # 需要 3 個 block，塞不下
    small = Sequence(seq_id="small", prompt_ids=list(range(2)), max_new_tokens=1)  # 需要 1 個 block，塞得下

    scheduler.add_request(big)
    scheduler.add_request(small)

    running = scheduler.step()
    assert running == []  # big 卡在隊伍最前面，small 不會被拿去插隊
    assert big.status == SequenceStatus.WAITING
    assert small.status == SequenceStatus.WAITING


# ---------------------------------------------------------------------------
# Scheduler：搶佔
# ---------------------------------------------------------------------------

def test_scheduler_preempts_running_sequence_when_another_needs_to_grow():
    bm = BlockManager(num_blocks=2, block_size=4)
    scheduler = Scheduler(bm)

    seq_a = Sequence(seq_id="a", prompt_ids=list(range(4)), max_new_tokens=10)  # 1 個 block
    seq_b = Sequence(seq_id="b", prompt_ids=list(range(4)), max_new_tokens=10)  # 1 個 block
    scheduler.add_request(seq_a)
    scheduler.add_request(seq_b)
    scheduler.step()  # 兩個都排進 running，剛好用滿 2 個 block

    assert set(s.seq_id for s in scheduler.running) == {"a", "b"}

    # 手動讓 a 的內容長大到需要第 2 個 block，逼出搶佔
    seq_a.output_ids = list(range(3))  # all_token_ids 長度變成 7，需要 2 個 block
    running = scheduler.step()

    running_ids = {s.seq_id for s in running}
    assert len(running_ids) == 1  # 只剩一個序列跑得動（池子只有 2 個 block）
    assert any("preempt" in e for e in scheduler.events)

    preempted_id = ({"a", "b"} - running_ids).pop()
    preempted_seq = seq_a if preempted_id == "a" else seq_b
    assert preempted_seq.status == SequenceStatus.WAITING
    assert preempted_seq.num_computed_tokens == 0
    assert bm.get_block_table(preempted_id) == []  # block 已歸還


def test_finish_frees_blocks_and_removes_from_running():
    bm = BlockManager(num_blocks=4, block_size=4)
    scheduler = Scheduler(bm)

    seq = Sequence(seq_id="s", prompt_ids=list(range(4)), max_new_tokens=1)
    scheduler.add_request(seq)
    scheduler.step()
    assert bm.num_free_blocks == 3

    scheduler.finish(seq)
    assert bm.num_free_blocks == 4
    assert seq not in scheduler.running


# ---------------------------------------------------------------------------
# 正確性驗證：交錯排程（含搶佔）不能改變任何序列的生成結果
# ---------------------------------------------------------------------------

def test_continuous_batching_matches_isolated_generation_even_with_preemption():
    """
    最重要的一個測試：多個序列在小記憶體池下交錯執行、發生搶佔，
    每個序列最終生成的文字，必須跟「它自己獨立、不受任何人干擾」
    跑出來的結果完全相同。這證明 recompute 式搶佔（丟 cache、
    之後整段重算）在數學上是無損的，不會因為排程的先後順序、
    有沒有被搶佔過而改變輸出。
    """
    tokenizer, config = make_tokenizer_and_config()
    model = TinyTransformerPaged(config)

    # 故意設一個很小的池子，逼出搶佔（跟 examples 腳本的思路一致）
    block_size = 4
    num_blocks = 6

    requests = [
        ("seq-A", "the quick ", 8, 0),
        ("seq-B", "the lazy dog ", 6, 0),
        ("seq-C", "fox jumps ", 8, 1),
    ]

    bm_batched = BlockManager(num_blocks=num_blocks, block_size=block_size)
    cache_batched = PagedKVCache(config, num_blocks=num_blocks, block_size=block_size)
    batched_results = run_continuous_batching(
        model, bm_batched, cache_batched, tokenizer, requests, verbose=False
    )

    # 每個序列各自獨立、用寬裕的池子跑一次，作為「正確答案」
    isolated_results = {}
    for seq_id, prompt, max_new, _ in requests:
        bm_iso = BlockManager(num_blocks=20, block_size=block_size)
        cache_iso = PagedKVCache(config, num_blocks=20, block_size=block_size)
        isolated_results[seq_id] = paged_generate(
            model, bm_iso, cache_iso, tokenizer, seq_id, prompt, max_new
        )

    assert batched_results == isolated_results


def test_all_sequences_eventually_finish_and_pool_fully_recovers():
    tokenizer, config = make_tokenizer_and_config()
    model = TinyTransformerPaged(config)

    block_size = 4
    num_blocks = 6
    bm = BlockManager(num_blocks=num_blocks, block_size=block_size)
    cache = PagedKVCache(config, num_blocks=num_blocks, block_size=block_size)

    requests = [
        ("seq-A", "the quick ", 5, 0),
        ("seq-B", "the lazy dog ", 5, 0),
        ("seq-C", "fox jumps ", 5, 2),
    ]
    results = run_continuous_batching(model, bm, cache, tokenizer, requests, verbose=False)

    assert set(results.keys()) == {"seq-A", "seq-B", "seq-C"}
    assert bm.num_free_blocks == num_blocks  # 全部做完後，池子應該完全恢復
