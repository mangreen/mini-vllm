"""
Stage 5：用 ChunkedPrefillScheduler 展示「長 prefill 不拖累 decode」。

情境設計：一個很長的 prompt（seq-long）跟兩個短 prompt（seq-a、
seq-b，主要在 decode）同時抵達。跟 Stage 3 的
`continuous_batching_generate.py` 對照著跑同一組請求：
  - Stage 3（不分 chunk）：seq-long 一旦被排進 running，會在單一
    一個 step 裡把它剩下的整段 prompt一次處理完。
  - Stage 5（chunked）：seq-long 的 prefill 被切成好幾個 step，
    每個 step 都把 budget 優先留給 seq-a、seq-b 的 decode。

觀察重點：seq-a、seq-b 拿到「每一個新 token」之間間隔了幾個
engine step——Stage 3 版本裡，只要 seq-long 剛好在跑，seq-a/seq-b
那一步都得等 seq-long 那次巨大的 forward 做完；Stage 5 版本裡，
seq-a/seq-b 幾乎每一步都能拿到新 token，不會被 seq-long 卡住。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.engine.chunked_scheduler import ChunkedPrefillScheduler
from mini_vllm.engine.sequence import Sequence
from mini_vllm.layers.paged_kv_cache import PagedKVCache
from mini_vllm.models.tiny_transformer import TinyTransformerConfig
from mini_vllm.models.tiny_transformer_paged import TinyTransformerPaged
from mini_vllm.models.tokenizer import CharTokenizer

torch.manual_seed(0)
torch.set_num_threads(4)


def run_chunked_prefill(
    model: TinyTransformerPaged,
    scheduler: ChunkedPrefillScheduler,
    block_manager: BlockManager,
    paged_cache: PagedKVCache,
    tokenizer: CharTokenizer,
    requests: list[tuple[str, str, int, int]],
    verbose: bool = False,
) -> tuple[dict[str, str], dict[str, list[float]]]:
    """
    跟 Stage 3 的 `run_continuous_batching` 幾乎一樣的骨架，差別
    只有兩處（都標了註解）：呼叫 `scheduler.step()` 拿到的是
    `(seq, num_tokens)` 配對，而且 forward 完之後要檢查
    `pending_token_count` 是否真的歸零，才可以取樣新 token
    ——chunked prefill 的中間步驟只是把一段很長的內容處理掉一部分，
    还沒看完整段內容前，不能提前用還不完整的資訊去猜下一個字。

    回傳：(每個序列生成的完整文字, 每個序列每個新 token『真正抵達的
    時間戳記』列表)。

    這裡刻意用 `time.perf_counter()` 記錄真實時間，而不是 engine
    step 的編號——因為我們的迴圈是單執行緒、同步執行的：一個 step
    裡不管排進了多少個序列、多少個 token，都要等這個 step 的所有
    forward 呼叫全部做完，才會進到下一個 step。如果某個 step 裡有
    一個序列的 chunk 特別大（一次處理很多 token），這個 step 本身
    花的『真實時間』就會被拉長，連帶讓下一個 step（以及裡面其他
    序列的 decode token）延後抵達——這正是 chunked prefill 想避免
    的事，而這個效果只有量測真實時間才看得出來，光看『第幾個
    step』是看不出來的（因為不管 chunk 多大，都只算『一個 step』）。
    """
    pending_arrivals = sorted(requests, key=lambda r: r[3])
    results: dict[str, str] = {}
    token_arrival_times: dict[str, list[float]] = {sid: [] for sid, *_ in requests}
    last_schedule_summary = None

    model.eval()
    step_idx = 0
    with torch.no_grad():
        while pending_arrivals or scheduler.has_unfinished_requests:
            while pending_arrivals and pending_arrivals[0][3] <= step_idx:
                seq_id, prompt, max_new, _ = pending_arrivals.pop(0)
                seq = Sequence(
                    seq_id=seq_id,
                    prompt_ids=tokenizer.encode(prompt),
                    max_new_tokens=max_new,
                )
                scheduler.add_request(seq)
                if verbose:
                    print(f"[step {step_idx:>3}] {seq_id} 抵達，加入 waiting queue")

            # 差異 1：這裡拿到的是 [(seq, num_tokens_this_step), ...]，
            # 不是單純的 running 列表。
            schedule = scheduler.step()

            if verbose and schedule:
                summary = ", ".join(f"{s.seq_id}:+{n}" for s, n in schedule)
                # 版本 B 裡 seq-long 常常連續好幾十步都是同一種排程組合
                # （例如一直是 seq-long:+32），逐行印出來只會洗版、
                # 沒有新資訊——同樣的組合只在第一次出現時印一次即可，
                # 觀察排程「有沒有變化」才是重點。
                if summary != last_schedule_summary:
                    print(f"[step {step_idx:>3}] 本步排程: {summary}")
                    last_schedule_summary = summary

            for seq, num_tokens in schedule:
                block_table = block_manager.get_block_table(seq.seq_id)
                input_ids, start_pos = seq.next_forward_input(max_tokens=num_tokens)
                logits = model(input_ids, paged_cache, start_pos=start_pos, block_table=block_table)
                seq.mark_computed(input_ids.shape[1])

                # 差異 2：只有這段內容真的被「看完」了（pending 歸零），
                # 才可以用最後一個位置的 logits 去猜下一個字——chunked
                # prefill 中途的某一步，input_ids 只是整段 prompt 的
                # 一小截，這一截的最後一個 token 不代表「目前已知內容
                # 的最後一個 token」，此時取樣沒有意義。
                if seq.pending_token_count == 0:
                    next_token = torch.argmax(logits[0, -1, :]).item()
                    seq.append_token(next_token)
                    token_arrival_times[seq.seq_id].append(time.perf_counter())

                    if seq.is_finished:
                        results[seq.seq_id] = tokenizer.decode(seq.all_token_ids)
                        if verbose:
                            preview = results[seq.seq_id]
                            if len(preview) > 60:
                                preview = preview[:60] + "...(略)"
                            print(f"[step {step_idx:>3}] {seq.seq_id} 完成: {preview!r}")
                        scheduler.finish(seq)

            step_idx += 1
            if step_idx > 10_000:
                raise RuntimeError("排程似乎卡死了，請檢查 ChunkedPrefillScheduler 邏輯")

    return results, token_arrival_times


def main() -> None:
    corpus = (
        "the quick brown fox jumps over the lazy dog runs sleeps plays "
        "and swims near the river abcdefghijklmnopqrstuvwxyz "
    )
    tokenizer = CharTokenizer(corpus)

    config = TinyTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=128,
        n_layers=2,
        n_heads=4,
        max_seq_len=2048,
    )
    model = TinyTransformerPaged(config)

    block_size = 4
    num_blocks = 600

    # 一個很長的 prompt（重複同一段語句拼接出來，模擬長文件/長對話
    # 歷史），跟兩個短 prompt（幾乎全程都在 decode）同時抵達。
    # 這裡刻意拼到接近 1000 個 token——實測這個模型在這台 CPU 上，
    # 單次 forward 的耗時會隨序列長度明顯增加（50 token 約 6ms，
    # 1000 token 約 290ms），長度太短的話，一次性 prefill 造成的
    # 延遲會被 Python/tensor 呼叫的固定開銷蓋過去，觀察不到效果。
    long_prompt = ("the quick brown fox jumps over the lazy dog " * 20).strip() + " "
    # 關鍵設計：seq-a、seq-b 先抵達，讓它們先完成自己的（短）prefill、
    # 進入穩定 decode 節奏之後，seq-long 才在第 5 步抵達——這樣才能
    # 真正驗證「decode 優先」這件事：如果三個序列在同一步一起抵達，
    # 大家都還在做各自的初次 prefill，『decode 優先』根本還沒有
    # 用武之地（因為此時沒有人是 decode_ready）。真實情境裡最常發生
    # 的也正是這種「已經在對話中的使用者，突然有新請求插進來」。
    requests = [
        ("seq-a", "runs and plays ", 10, 0),
        ("seq-b", "sleeps near the river ", 10, 0),
        ("seq-long", long_prompt, 3, 5),
    ]

    print(f"seq-long 的 prompt 長度: {len(tokenizer.encode(long_prompt))} 個 token\n")

    # --- 版本 A：Stage 3 風格，不分 chunk（用一個「budget 無限大」的
    #     ChunkedPrefillScheduler 模擬——等同於 Stage 3 原本的行為）---
    print("=== 版本 A：不分 chunk（一次把整段 prefill 做完）===")
    bm_a = BlockManager(num_blocks=num_blocks, block_size=block_size)
    cache_a = PagedKVCache(config, num_blocks=num_blocks, block_size=block_size)
    scheduler_a = ChunkedPrefillScheduler(
        bm_a, token_budget_per_step=10_000, max_prefill_chunk=10_000
    )
    results_a, arrivals_a = run_chunked_prefill(
        model, scheduler_a, bm_a, cache_a, tokenizer, requests, verbose=True
    )

    print("\n=== 版本 B：Chunked Prefill（decode 優先，長 prefill 切塊）===")
    bm_b = BlockManager(num_blocks=num_blocks, block_size=block_size)
    cache_b = PagedKVCache(config, num_blocks=num_blocks, block_size=block_size)
    scheduler_b = ChunkedPrefillScheduler(bm_b, token_budget_per_step=36, max_prefill_chunk=32)
    results_b, arrivals_b = run_chunked_prefill(
        model, scheduler_b, bm_b, cache_b, tokenizer, requests, verbose=True
    )

    print("\n=== 正確性檢查 ===")
    print(f"兩個版本生成結果是否完全相同: {results_a == results_b}")

    print("\n=== decode 節奏比較：seq-a 相鄰 token 之間的真實間隔（毫秒）===")
    gaps_a = [
        (b - a) * 1000 for a, b in zip(arrivals_a["seq-a"], arrivals_a["seq-a"][1:])
    ]
    gaps_b = [
        (b - a) * 1000 for a, b in zip(arrivals_b["seq-a"], arrivals_b["seq-a"][1:])
    ]
    print(f"版本 A（不分 chunk）: {[f'{g:.1f}' for g in gaps_a]}")
    print(f"版本 B（chunked）   : {[f'{g:.1f}' for g in gaps_b]}")
    print(f"版本 A 最大間隔: {max(gaps_a):.1f} ms，平均間隔: {sum(gaps_a)/len(gaps_a):.1f} ms")
    print(f"版本 B 最大間隔: {max(gaps_b):.1f} ms，平均間隔: {sum(gaps_b)/len(gaps_b):.1f} ms")
    print(
        "→ 版本 A 裡，seq-long 的整段 prefill 是在『某一個 step』裡一次做完的，"
        "\n   那個 step 的真實耗時被拉得特別長，連帶讓下一個 step 開始的時間延後，"
        "\n   seq-a 在那附近會出現一次明顯的延遲尖峰（最大間隔）；"
        "\n   版本 B 因為 budget 有上限、prefill 被切成小塊，每個 step 的真實耗時"
        "\n   都被控制在差不多的範圍內，seq-a 的間隔更平均、沒有尖峰。"
    )


if __name__ == "__main__":
    main()
