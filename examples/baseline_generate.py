"""
Stage 0：樸素 baseline —— 完全不做任何快取優化的生成迴圈。

這支腳本刻意「笨」：每一個 decode step 都把目前為止的完整
input_ids 重新丟進模型跑一次 forward，即使前面的 token
在上一個 step 早就算過一模一樣的東西。

目的：
  1. 先確保生成邏輯（貪婪取樣 + 逐 token 生成）是對的，
     作為之後所有優化版本的正確性對照組。
  2. 實際量測、畫出「每個 step 花的時間」，親眼看見它隨序列
     長度增加而變慢（O(n^2) 的來源），而不是只在腦中想像。

ex.
目標：輸入 "a"，生成 2 個新 token。

第 1 圈：
  - 丟入: "a"
  - 大腦計算: 重頭算 "a" 的特徵 ──> 預測下一個字是 "b"
  - 拼接: "ab"

第 2 圈：
  - 丟入: "ab"
  - 大腦計算: 【重複算 "a"】+ 重頭算 "b" ──> 預測下一個字是 "c"  <-- 重複計算產生！
  - 拼接: "abc"

結束，回傳 "abc" 與每圈耗時

```python
prompt = "a"
# naive_generate 會跑迴圈：
# Step 1: 餵入 "a"     -> 拿到 logits -> 取最後位置最大值 -> 得 "b"
# Step 2: 餵入 "ab"    -> 重新計算全部 -> 取最後位置最大值 -> 得 "c"
# 回傳 ("abc", [0.003, 0.004])
```
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_vllm.models.tiny_transformer import TinyTransformer, TinyTransformerConfig
from mini_vllm.models.tokenizer import CharTokenizer

# 固定 CPU thread 數與 random seed，讓每次跑的結果、計時可重現、可比較
torch.manual_seed(0)
torch.set_num_threads(4)


def naive_generate(
    model: TinyTransformer,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, list[float]]:
    """
    最樸素的生成迴圈：每個 step 都把「完整序列」重新餵給模型。

    回傳：
      - 生成後的完整字串
      - 每個 step 花費的秒數列表（用來畫出時間隨長度成長的曲線）
    """
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)  # [1, T]

    step_times: list[float] = []

    # 這裡用 torch.no_grad()，避免計算梯度、節省記憶體
    with torch.no_grad():
        for _ in range(max_new_tokens):
            t0 = time.perf_counter()

            # 重點：這裡永遠丟「目前累積的整個序列」進去，
            # 模型內部會重新計算所有 token 的 attention，
            # 即使前面的 token 早就算過完全一樣的結果。
            logits = model(input_ids)  # [1, T, vocab_size]

            step_times.append(time.perf_counter() - t0)

            next_token_logits = logits[0, -1, :]  # 只需要最後一個位置的預測
            next_token = torch.argmax(next_token_logits).item()  # 貪婪取樣

            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]], dtype=torch.long)], dim=1
            )

    generated_ids = input_ids[0].tolist()
    return tokenizer.decode(generated_ids), step_times


def main() -> None:
    # 小語料，字元級 vocab；模型完全沒訓練過（random init），
    # Stage 0 的重點是「生成迴圈的機制」，不是「生成的內容有沒有意義」。

    # 建立字元級 tokenizer
    corpus = "the quick brown fox jumps over the lazy dog abcdefghijklmnopqrstuvwxyz "
    tokenizer = CharTokenizer(corpus)

    # 建立小模型
    config = TinyTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=128,
        n_layers=2,
        n_heads=4,
        max_seq_len=256,
    )
    model = TinyTransformer(config)

    # 設定 prompt 與要生成的 token 數量
    prompt = "the quick "
    max_new_tokens = 40

    # 生成
    text, step_times = naive_generate(model, tokenizer, prompt, max_new_tokens)

    print(f"Prompt: {prompt!r}")
    print(f"生成結果（模型未訓練，內容本身無意義，重點是流程跟計時）：\n{text!r}\n")

    print(f"{'step':>5} | {'目前長度':>8} | {'耗時 (ms)':>8} | {'目前內容':>20}")
    print("-" * 36)

    # 印出每個 step 的耗時，並說明 O(n^2) 的原因
    for i, t in enumerate(step_times):
        seq_len = len(prompt) + i + 1
        print(f"{i:>5} | {seq_len:>12} | {t * 1000:>10.3f} | {text[:i+1]!r}")

    avg_first_half = sum(step_times[: len(step_times) // 2]) / (len(step_times) // 2)
    avg_second_half = sum(step_times[len(step_times) // 2 :]) / (
        len(step_times) - len(step_times) // 2
    )

    # 印出前半段與後半段的平均耗時，並說明 O(n^2) 的原因
    print(
        f"\n前半段平均耗時: {avg_first_half * 1000:.3f} ms, "
        f"後半段平均耗時: {avg_second_half * 1000:.3f} ms"
    )
    print(
        "→ 序列變長後，每個 step 的耗時應該要有成長趨勢（O(n^2) 的樸素做法），"
        "\n   這就是 Stage 1 要引入 KV cache 解決的問題。"
        "\n   註：這個小模型在 CPU 上單一 step 的計算量太小，"
        "\n   成長趨勢可能被系統雜訊蓋過，重點是理解「為什麼」而不是這次量到的絕對數字。"
    )


if __name__ == "__main__":
    main()
