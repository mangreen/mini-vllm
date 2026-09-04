"""
Stage 3：用 Scheduler 跑 continuous batching，模擬多個請求交錯抵達、
交錯完成的情境。

跟 Stage 2 的 `paged_attention_generate.py` 對照著看差異最清楚：
  - Stage 2：一個序列從頭跑到尾生成完，才換下一個序列開始
            （序列之間是「先後」共用 block 池，不是「同時」）。
  - Stage 3：每個 engine step 都重新排程一次。序列可以在中途加入
            （新請求抵達）、中途離開（生成完成），甚至中途被踢出去
            又重新排回來（記憶體不夠時的搶佔）。

本階段的模型 forward 呼叫仍然是「每個序列各自呼叫一次」（見下面
`run_continuous_batching` 迴圈裡的 `for seq in running`），還沒有
真正把多個序列的 tensor 疊在一起、用一次 forward 呼叫算完整批——
那需要處理「不同序列長度不同、block table 不同」時如何 padding/
mask 的問題，本身是不小的工程量，也是 vLLM 真正的 CUDA kernel
（varlen attention）在解決的事。**本階段展示的是「排程政策」這個
層次的 continuous batching**——動態准入、動態搶佔、序列交錯執行——
這正是 continuous batching 這個詞最早被提出時指的東西；把多個序列
的 forward 融合成一次 kernel 呼叫，是更底層的效能優化，留給
Stage 6。

---

1. 傳統 Batching（Static / Traditional Batching）:
傳統的做法是以「一批序列」為單位。必須等整批中最長、最慢的序列生成結束，
CPU/GPU 才能清空記憶體接下一批。這會產生大量的 Bubble（無效等待與 Padding 浪費）。

時間軸 (Engine Steps) ───>
Step    0   1   2   3   4   5   6   7   8   9  10
      ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
Seq A │ P │ D │ D │ D │ D │ D │ D │ D │ D │ D │ D │ (最慢，長度 10)
      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
Seq B │ P │ D │ D │░░░│░░░│░░░│░░░│░░░│░░░│░░░│░░░│ (Step 2 就完成了，但要乾等！)
      └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
                                                  ▲
                                                  └─── 到了 Step 10 才能全部釋放，
                                                       新請求 Seq C 才能開始！
註：P = Prefill (處理 Prompt), D = Decode (生成 1 個 token), ░ = Idle / Bubble (浪費時間與記憶體)

---

2. Continuous Batching（Iteration-level Scheduling）:
核心在於：每一個 Engine Step 都重新排程一次。

時間軸 (Engine Steps) ───>
step_idx=0      step_idx=2
        ↓ ────> ↓
Step    0   1   2   3   4   5   6   7   8   9  10
      ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
Seq A │ P │ D │ D │ D │ D │ D │ D │ D │ D │ D │ D │ (持續運行)
      ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
Seq B │ P │ D │ D │ 釋放 ! (Finish)
      └───┴───┴───┼───┬───┬───┬───┬───┐
Seq C           ▲ │ P │ D │ D │ D │ D │ (Step 3 動態加入，不需要乾等！)
                │ └───┴───┴───┴───┴───┘    
(Seq C 在 Step 2 抵達)          
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.engine.scheduler import Scheduler
from mini_vllm.engine.sequence import Sequence
from mini_vllm.layers.paged_kv_cache import PagedKVCache
from mini_vllm.models.tiny_transformer import TinyTransformerConfig
from mini_vllm.models.tiny_transformer_paged import TinyTransformerPaged
from mini_vllm.models.tokenizer import CharTokenizer

torch.manual_seed(0)
torch.set_num_threads(4)


def run_continuous_batching(
    model: TinyTransformerPaged,
    block_manager: BlockManager,
    paged_cache: PagedKVCache,
    tokenizer: CharTokenizer,
    requests: list[tuple[str, str, int, int]], # (seq_id, prompt, max_new_tokens, arrival_step)
    verbose: bool = False,
) -> dict[str, str]:
    """
    requests: (seq_id, prompt, max_new_tokens, arrival_step) 的列表。
      arrival_step 決定這個請求要等到第幾個 engine step 才會抵達，
      藉此模擬「請求不是同時到齊」的真實情境。

    回傳：{seq_id: 生成的完整文字}
    """
    scheduler = Scheduler(block_manager)
    pending_arrivals = sorted(requests, key=lambda r: r[3]) # 按 arrival_step 排序，方便每個 step 只看最前面幾個
    results: dict[str, str] = {} # {seq_id: 生成的完整文字}

    model.eval()
    step_idx = 0 
    with torch.no_grad(): # Stage 3 的模型 forward 呼叫仍然是「每個序列各自呼叫一次」，還沒有真正把多個序列的 tensor 疊在一起、用一次 forward 呼叫算完整批。
        # 這個 while loop 的條件是「還有請求沒抵達，或是 scheduler 還有未完成的請求」。
        while pending_arrivals or scheduler.has_unfinished_requests:
            # 新請求抵達，排進 waiting 隊伍
            # 這裡用 while 而不是 if，因為可能有多個請求同時抵達同一個 step
            while pending_arrivals and pending_arrivals[0][3] <= step_idx:
                seq_id, prompt, max_new, _ = pending_arrivals.pop(0)
                seq = Sequence(
                    seq_id=seq_id,
                    prompt_ids=tokenizer.encode(prompt),
                    max_new_tokens=max_new,
                )
                scheduler.add_request(seq)
                if verbose:
                    print(f"[step {step_idx:>2}] {seq_id} 抵達，加入 waiting queue")

            # 這個 step 的排程決策：哪些序列可以加入 running？哪些序列要被搶佔？
            # 這裡的 step() 呼叫會修改 scheduler.running 的內容，並在 scheduler.events 裡記錄「准入、搶佔」事件。
            running_before = {s.seq_id for s in scheduler.running}
            running = scheduler.step()
            running_after = {s.seq_id for s in running}

            if verbose:
                for event in scheduler.events:
                    print(f"[step {step_idx:>2}] {event}")
                newly_admitted = running_after - running_before
                if newly_admitted and not any("admit" in e for e in scheduler.events):
                    # 理論上 admit 事件已經涵蓋這個資訊，這裡是防呆備援
                    print(f"[step {step_idx:>2}] 新加入 running: {sorted(newly_admitted)}")

            # 這個 step 的 forward 計算：
            # 對每個 running 裡的序列，呼叫模型 forward，生成下一個 token。
            # 這裡的 for 迴圈是「每個序列各自呼叫一次 forward」，還沒有真正把多個序列的 tensor 疊在一起、用一次 forward 呼叫算完整批。
            # 這裡的 for 迴圈裡，seq.next_forward_input() 會回傳「這個序列下一個 forward 要丟進模型的 input_ids」，
            # 以及「這個 input_ids 對應到 KV cache 的起始位置 start_pos」。
            # 這個方法會自動計算出「目前還沒反映進 cache 的那一段」的 token，
            # 無論是初次 prefill、正常 decode，還是被搶佔後重新排進 running 都適用。
            for seq in list(running):
                if not seq.needs_forward:
                    continue
                input_ids, start_pos = seq.next_forward_input()
                block_table = block_manager.get_block_table(seq.seq_id)
                logits = model(input_ids, paged_cache, start_pos=start_pos, block_table=block_table)
                seq.mark_computed(input_ids.shape[1])

                next_token = torch.argmax(logits[0, -1, :]).item()
                seq.append_token(next_token)

                if seq.is_finished:
                    results[seq.seq_id] = tokenizer.decode(seq.all_token_ids)
                    if verbose:
                        print(f"[step {step_idx:>2}] {seq.seq_id} 完成: {results[seq.seq_id]!r}")
                    scheduler.finish(seq)

            step_idx += 1 # 下一個 engine step
            if step_idx > 10_000:
                raise RuntimeError("排程似乎卡死了，請檢查 Scheduler 邏輯")

    return results


def main() -> None:
    corpus = "the quick brown fox jumps over the lazy dog abcdefghijklmnopqrstuvwxyz "
    tokenizer = CharTokenizer(corpus)

    config = TinyTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=128,
        n_layers=2,
        n_heads=4,
        max_seq_len=256,
    )
    model = TinyTransformerPaged(config)

    # 刻意設一個偏小的池子：三個序列同時跑，會逼近容量上限，
    # 這樣才有機會實際示範到「搶佔」發生的樣子。
    block_size = 4
    num_blocks = 8

    block_manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
    paged_cache = PagedKVCache(config, num_blocks=num_blocks, block_size=block_size)

    requests = [
        ("seq-A", "the quick ", 10, 0),   # 第 0 步就抵達
        ("seq-B", "the lazy dog ", 8, 0),  # 也是第 0 步抵達
        ("seq-C", "fox jumps over ", 10, 2),  # 晚 2 步才抵達，展示動態加入
    ]

    print("=== Continuous Batching 排程展示 ===")
    print(f"block_size={block_size}, num_blocks={num_blocks}\n")

    results = run_continuous_batching(
        model, block_manager, paged_cache, tokenizer, requests, verbose=True
    )

    print("\n=== 最終生成結果 ===")
    for seq_id, text in results.items():
        print(f"{seq_id}: {text!r}")


if __name__ == "__main__":
    main()
