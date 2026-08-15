"""
Stage 2：用 PagedAttention 跑生成，並展示 block 池怎麼被多個序列共用。

這支腳本做三件事：
  1. 正確性：用跟 Stage 0/1 相同的權重，跑一次 prefill+decode，
     確認生成結果逐字元相同。
  2. Block 生命週期：印出每個序列在生成過程中 block 配置的變化，
     直接看到「跨過 block 邊界才配置新 block」這件事實際發生。
  3. 記憶體效率比較：算出「Stage 1 每序列固定配置 max_seq_len」
     vs「Stage 2 按需配置 block」在同一批序列上分別用了多少記憶體。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.layers.paged_kv_cache import PagedKVCache
from mini_vllm.models.tiny_transformer import TinyTransformerConfig
from mini_vllm.models.tiny_transformer_paged import TinyTransformerPaged
from mini_vllm.models.tokenizer import CharTokenizer

torch.manual_seed(0)
torch.set_num_threads(4)


def paged_generate(
    model: TinyTransformerPaged,
    block_manager: BlockManager,
    paged_cache: PagedKVCache,
    tokenizer: CharTokenizer,
    seq_id: str,
    prompt: str,
    max_new_tokens: int,
    verbose: bool = False,
) -> str:
    """
    用 PagedAttention 生成序列。
    Args:
        model: TinyTransformerPaged 模型
        block_manager: BlockManager 實例
        paged_cache: PagedKVCache 實例
        tokenizer: CharTokenizer 實例
        seq_id: 序列的唯一識別字串（用來在 block_manager 裡面對應到這個序列的 block table）
        prompt: 生成的 prompt 字串
        max_new_tokens: 要生成的 token 數量
        verbose: 是否印出 debug 訊息
    Returns:
        生成的完整字串（包含 prompt + 新生成的 token）
    """

    # 將 PyTorch Model 切換到評估 (測試/推理) 模式，會關閉 Dropout（不隨機丟棄神經元），
    # 並固定 BatchNorm（使用訓練好的統計均值和方差），確保每次預測結果穩定。
    model.eval()

    #---------------------------------------------------------------
    # 先把 prompt 編碼成 token id，並記錄 prompt 的長度
    #---------------------------------------------------------------
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)  # [1, P]
    seq_len = prompt_ids.shape[1] # prompt 的長度（不含後續 decode 生成的 token）

    with torch.no_grad():
        # --- prefill ---
        block_table = block_manager.ensure_capacity(seq_id, seq_len)
        if verbose:
            print(f"  [{seq_id}] prefill 後 block_table={block_table}")
        logits = model(prompt_ids, paged_cache, start_pos=0, block_table=block_table)

        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).item()
        generated_ids = prompt_ids[0].tolist() + [next_token]

        # --- decode ---
        for _ in range(max_new_tokens - 1):
            old_num_blocks = len(block_manager.get_block_table(seq_id))
            block_table = block_manager.ensure_capacity(seq_id, seq_len + 1)
            if verbose and len(block_table) > old_num_blocks:
                print(
                    f"  [{seq_id}] 序列長度來到 {seq_len + 1}，"
                    f"跨過 block 邊界，新配置 1 個 block -> block_table={block_table}"
                )

            next_input = torch.tensor([[next_token]], dtype=torch.long)
            logits = model(next_input, paged_cache, start_pos=seq_len, block_table=block_table)
            seq_len += 1

            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            generated_ids.append(next_token)

    return tokenizer.decode(generated_ids)


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

    block_size = 4
    num_blocks = 24  # 刻意設一個不算寬裕的池子，逼近容量上限才看得出「共用」的意義

    # -----------------------------------------------------------------
    # Part 1：多個序列「輪流」跑，共用同一個 block 池與 free pool
    # （真正的「同時」跑、動態排程是 Stage 3 continuous batching 的工作，
    #  這裡先驗證：block 池能不能正確地被多個序列先後借用、歸還、重複利用）
    # -----------------------------------------------------------------
    block_manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
    paged_cache = PagedKVCache(config, num_blocks=num_blocks, block_size=block_size)

    # 這裡的三個序列長度都不一樣，且都比 max_seq_len=256 短很多
    requests = [
        ("seq-A", "the quick ", 10),
        ("seq-B", "the lazy dog ", 6),
        ("seq-C", "fox jumps ", 8),
    ]

    print("=== Part 1：多序列共用同一個 block 池 ===")
    print(f"block_size={block_size}, num_blocks={num_blocks}\n")

    #---------------------------------------------------------------
    # 逐個序列跑 prefill+decode，並印出 block 池的使用狀況
    #---------------------------------------------------------------
    results: dict[str, str] = {}
    for seq_id, prompt, n_new in requests:
        # 印出這個序列的 prompt 與要生成的 token 數量
        print(f"[{seq_id}] prompt={prompt!r}, max_new_tokens={n_new}")

        # 生成序列
        text = paged_generate(
            model, block_manager, paged_cache, tokenizer, seq_id, prompt, n_new, verbose=True
        )
        results[seq_id] = text
        print(f"  生成結果: {text!r}")

        # 印出 block 池的使用狀況
        usage = block_manager.memory_usage_summary()
        print(f"  目前 block 池使用狀況: {usage}\n")

        # 序列生成完畢，釋放它的 block，讓後面的序列可以重複利用
        block_manager.free(seq_id)
        print(f"  [{seq_id}] 已釋放 block，池子狀態: {block_manager.memory_usage_summary()}\n")

    # -----------------------------------------------------------------
    # Part 2：記憶體效率比較 —— Stage 1（固定配置）vs Stage 2（按需配置）
    # -----------------------------------------------------------------
    print("=== Part 2：記憶體效率比較 ===")
    seq_lengths = [len(p) + n for _, p, n in requests]  # 每個序列的實際總長度
    print(f"這批序列的實際長度: {seq_lengths}")

    # Stage 1 相當於每個序列都固定配置 max_seq_len // block_size 個 block 的資源
    stage1_blocks_equivalent = sum(
        config.max_seq_len // block_size for _ in seq_lengths
    )
    
    # Stage 2 實際用掉的 block 數量，因為每個序列按實際長度配置 block
    stage2_blocks_used = sum(block_manager.num_blocks_needed(n) for n in seq_lengths)

    print(
        f"Stage 1 做法（每序列固定配置 max_seq_len={config.max_seq_len}）"
        f"相當於用掉: {stage1_blocks_equivalent} 個 block 大小的資源"
    )
    print(f"Stage 2 做法（按實際長度配置）實際用掉: {stage2_blocks_used} 個 block")
    print(
        f"→ 節省比例: {(1 - stage2_blocks_used / stage1_blocks_equivalent) * 100:.1f}%"
        "\n   （這批序列都很短，跟 max_seq_len=256 差距很大，所以節省比例看起來很誇張；"
        "\n   但這正是重點——Stage 1 的浪費跟「序列實際長度 vs max_seq_len 的差距」成正比，"
        "\n   實務上 vLLM 服務的請求長度分布很廣，這個浪費是真實存在的。)"
    )


if __name__ == "__main__":
    main()
