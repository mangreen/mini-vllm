"""
Stage 1：用 KV Cache 做生成，並跟 Stage 0 的樸素版本比較。

流程分成兩段：
  1. prefill：把 prompt 一次性丟進模型，寫滿 KV cache 的前 N 個位置。
  2. decode：每次只丟「上一步生成的 1 個新 token」進模型，
     模型內部只需要算這 1 個新 token 的 Q/K/V，
     attention 則是拿這 1 個 Q 去對「快取裡全部的 K/V」做運算。

跟 examples/baseline_generate.py 對照著看效果最好：
  - 兩者在貪婪取樣下，生成結果必須逐字元相同（本檔案最後會自動驗證）。
  - baseline 版本每個 step 都要重新算「全部」token 的 Q/K/V + attention；
    這裡每個 decode step 只算「新 token」自己的 Q/K/V + 對快取做 attention，
    理論上每個 decode step 的計算量應該明顯比 baseline 版本少、
    也更不會隨著序列變長而爆炸性成長。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_vllm.layers.kv_cache import KVCache
from mini_vllm.models.tiny_transformer import TinyTransformerConfig
from mini_vllm.models.tiny_transformer_kv import TinyTransformerKV
from mini_vllm.models.tokenizer import CharTokenizer

torch.manual_seed(0)
torch.set_num_threads(4)


def kv_cache_generate(
    model: TinyTransformerKV,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, list[float]]:
    """
    回傳：
      - 生成後的完整字串
      - 每個 step（含 prefill 那一步）花費的秒數列表，
        方便跟 baseline 版本的 step_times 對照著畫圖比較。
    """
    model.eval()
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)  # [1, P]

    cache = KVCache(model.config, batch_size=1)
    step_times: list[float] = []

    with torch.no_grad():
        # --- prefill：一次處理整個 prompt ---
        # 這裡的 start_pos=0，代表 prompt 的第一個 token 對應到快取的第 0 個位置。
        t0 = time.perf_counter() # --- 記錄 prefill 開始時間 ---
        logits = model(prompt_ids, kv_cache=cache, start_pos=0) # --- 把 prompt 丟進模型，得到 logits ---
        cache.advance(prompt_ids.shape[1]) # --- 把快取長度往前推 prompt 長度，代表下一個新 token 對應到快取的下一個位置 ---
        step_times.append(time.perf_counter() - t0) # --- 記錄 prefill 耗時 ---

        # --- 取樣第一個新 token ---
        next_token_logits = logits[0, -1, :] # 取最後一個位置的 logits
        next_token = torch.argmax(next_token_logits).item() # 取最大值的索引作為下一個 token
        generated_ids = prompt_ids[0].tolist() + [next_token] # 把 prompt 的 token 加上第一個新 token，作為生成序列的初始值

        # --- decode：每次只丟 1 個新 token ---
        for _ in range(max_new_tokens - 1):
            # 這裡的 start_pos=cache.length，代表「這個新 token」對應到快取的第 cache.length 個位置。
            t0 = time.perf_counter() # --- 記錄 decode step 開始時間 ---
            next_input = torch.tensor([[next_token]], dtype=torch.long)  # [1, 1]
            logits = model(next_input, kv_cache=cache, start_pos=cache.length) # --- 把新 token 丟進模型，得到 logits ---
            cache.advance(1) # --- 把快取長度往前推 1，代表下一個新 token 對應到快取的下一個位置 ---
            step_times.append(time.perf_counter() - t0) # --- 記錄 decode step 耗時 ---

            # --- 取樣下一個新 token ---
            next_token_logits = logits[0, -1, :] # 取最後一個位置的 logits
            next_token = torch.argmax(next_token_logits).item() # 取最大值的索引作為下一個 token
            generated_ids.append(next_token) # 把新 token 加到生成序列中

    return tokenizer.decode(generated_ids), step_times


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
    model = TinyTransformerKV(config)

    # 這裡的 prompt 可以任意改，max_new_tokens 也可以改。
    prompt = "the quick "
    max_new_tokens = 40

    # 生成結果 + 計時
    text, step_times = kv_cache_generate(model, tokenizer, prompt, max_new_tokens)

    print(f"Prompt: {prompt!r}")
    print(f"生成結果（模型未訓練，內容本身無意義，重點是流程跟計時）：\n{text!r}\n")

    print(f"{'step':>5} | {'類型':>8} | {'目前序列長度':>12} | {'耗時 (ms)':>10}")
    print("-" * 46)
    for i, t in enumerate(step_times):
        kind = "prefill" if i == 0 else "decode"
        seq_len = len(prompt) if i == 0 else len(prompt) + i
        print(f"{i:>5} | {kind:>8} | {seq_len:>12} | {t * 1000:>10.3f}")

    decode_times = step_times[1:]
    avg_first_half = sum(decode_times[: len(decode_times) // 2]) / (len(decode_times) // 2)
    avg_second_half = sum(decode_times[len(decode_times) // 2 :]) / (
        len(decode_times) - len(decode_times) // 2
    )
    print(
        f"\ndecode 前半段平均耗時: {avg_first_half * 1000:.3f} ms, "
        f"後半段平均耗時: {avg_second_half * 1000:.3f} ms"
    )
    print(
        "→ 跟 baseline_generate.py 的結果對照：這裡每個 decode step 只需要算"
        "\n   1 個新 token 的 Q/K/V + 對 cache 做 attention，理論計算量是 O(目前長度)，"
        "\n   而不是 baseline 版本每步都要重算全部 token 的 O(目前長度^2)。"
        "\n   在這個小模型、小序列長度下，兩者的絕對時間可能差異不大"
        "\n  （常數項、系統雜訊主導），但隨著模型變大、序列變長，"
        "\n   這個差異會越來越明顯——這正是 KV cache 存在的意義。"
    )


if __name__ == "__main__":
    main()
