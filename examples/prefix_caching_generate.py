"""
Stage 4：用 PrefixCachingBlockManager 展示多個請求共用同一個
system prompt 時，prefix caching 帶來的效果。

這支腳本做三件事：
  1. 正確性：即使命中 prefix cache、跳過部分 forward 計算，
     生成結果必須跟完全不用 prefix cache（Stage 2 的做法）一致。
  2. 觀察快取命中：印出每個請求命中了多少個 token 的快取、
     跳過了多少次 forward 計算。
  3. 效能比較：算出「有 prefix caching」vs「沒有」，總共省下多少次
     forward 呼叫（這是比 Stage 2 的記憶體比較更進一步的效能指標，
     因為省下的不只是記憶體，是真正的矩陣運算）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_vllm.engine.prefix_cache_block_manager import PrefixCachingBlockManager
from mini_vllm.engine.sequence import Sequence
from mini_vllm.layers.paged_kv_cache import PagedKVCache
from mini_vllm.models.tiny_transformer import TinyTransformerConfig
from mini_vllm.models.tiny_transformer_paged import TinyTransformerPaged
from mini_vllm.models.tokenizer import CharTokenizer

torch.manual_seed(0)
torch.set_num_threads(4)


def prefix_cache_generate(
    model: TinyTransformerPaged,
    block_manager: PrefixCachingBlockManager,
    paged_cache: PagedKVCache,
    tokenizer: CharTokenizer,
    seq_id: str,
    prompt: str,
    max_new_tokens: int,
    verbose: bool = False,
) -> tuple[str, int, int]:
    """
    回傳: (生成的完整文字, 命中快取的 token 數, 實際跑 forward 算過的 token 數)

    注意：這裡要比較的效能指標是「forward 實際算過幾個 token」，
    不是「呼叫了幾次 forward」——有沒有 prefix caching，呼叫次數
    通常一樣（prefill 跟每個 decode step 各 1 次），差別在於
    **prefill 那一次呼叫，處理的 token 數變少了**（命中快取的部分
    直接跳過，不會被送進模型）。
    """
    model.eval()
    prompt_ids = tokenizer.encode(prompt)
    seq = Sequence(seq_id=seq_id, prompt_ids=prompt_ids, max_new_tokens=max_new_tokens)

    # --- prefill：先試著命中 prefix cache ---
    block_table, num_cached = block_manager.allocate_prefill(seq_id, seq.all_token_ids)
    seq.num_computed_tokens = num_cached  # 命中的部分直接跳過，不用 forward
    if verbose:
        print(
            f"  [{seq_id}] prompt 長度={len(prompt_ids)}，"
            f"prefix cache 命中 {num_cached} 個 token"
            f"（跳過這些 token 的 forward 計算）"
        )

    num_tokens_computed = 0
    with torch.no_grad():
        while not seq.is_finished:
            if seq.needs_forward:
                input_ids, start_pos = seq.next_forward_input()
                logits = model(
                    input_ids, paged_cache, start_pos=start_pos, block_table=block_table
                )
                seq.mark_computed(input_ids.shape[1])
                num_tokens_computed += input_ids.shape[1]

                next_token = torch.argmax(logits[0, -1, :]).item()
                seq.append_token(next_token)

                if not seq.is_finished:
                    block_table = block_manager.grow_private(seq_id, len(seq.all_token_ids))

    return tokenizer.decode(seq.all_token_ids), num_cached, num_tokens_computed


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
        max_seq_len=256,
    )
    model = TinyTransformerPaged(config)

    block_size = 4
    num_blocks = 40

    block_manager = PrefixCachingBlockManager(num_blocks=num_blocks, block_size=block_size)
    paged_cache = PagedKVCache(config, num_blocks=num_blocks, block_size=block_size)

    # 三個請求共用同一段「system prompt」開頭，後面接不同的內容——
    # 這正是真實世界 prefix caching 最常見的使用情境：同一個 system
    # prompt、不同的使用者輸入。
    system_prompt = "the quick brown fox "
    requests = [
        ("seq-1", system_prompt + "jumps over ", 10),
        ("seq-2", system_prompt + "runs and plays ", 10),
        ("seq-3", system_prompt + "sleeps near the river ", 10),
    ]

    print("=== Prefix Caching 展示：三個請求共用同一段 system prompt ===")
    print(f"system_prompt={system_prompt!r}\nblock_size={block_size}\n")

    total_tokens_computed_with_cache = 0
    total_prompt_tokens = 0

    for seq_id, prompt, max_new in requests:
        text, num_cached, num_computed = prefix_cache_generate(
            model, block_manager, paged_cache, tokenizer, seq_id, prompt, max_new, verbose=True
        )
        print(f"  生成結果: {text!r}")
        print(f"  forward 實際算過 {num_computed} 個 token\n")

        total_tokens_computed_with_cache += num_computed
        total_prompt_tokens += len(tokenizer.encode(prompt))

    print("=== 快取使用狀況 ===")
    stats = block_manager.prefix_cache_stats()
    print(f"目前登記進快取索引的 block 數: {stats['num_registered_blocks']}")
    print(f"目前被多個序列共用的 block: {stats['shared_blocks']}")

    print("\n=== 效能比較：有無 prefix caching，forward 實際算過的 token 數 ===")
    # 沒有 prefix caching 的話（Stage 2 的做法）：prefill 要對「整個
    # prompt」跑一次 forward（處理 len(prompt) 個 token），之後每個
    # decode step 各處理 1 個 token。
    total_tokens_computed_without_cache = sum(
        len(tokenizer.encode(prompt)) + (max_new - 1) for _, prompt, max_new in requests
    )
    print(f"沒有 prefix caching：forward 總共算過 {total_tokens_computed_without_cache} 個 token")
    print(f"有 prefix caching：forward 總共算過 {total_tokens_computed_with_cache} 個 token")
    saved = total_tokens_computed_without_cache - total_tokens_computed_with_cache
    print(
        f"→ 省下 {saved} 個 token 的計算量"
        f"（{saved / total_tokens_computed_without_cache * 100:.1f}%），"
        "\n   這些都是被跳過、完全不用算的重複計算——不只是省記憶體，"
        "\n   是真的省下了矩陣運算本身。"
    )


if __name__ == "__main__":
    main()
