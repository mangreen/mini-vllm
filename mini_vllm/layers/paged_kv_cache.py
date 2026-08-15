"""
Stage 2：PagedKVCache —— 實際存放 K/V 數值的「實體 block 池」。

跟 BlockManager 的分工：
  - `BlockManager`（engine/block_manager.py）：只管「帳」，決定每個
    序列的邏輯 block 對應到池子裡哪個實體 block 編號，完全不碰
    任何 tensor 數值。
  - `PagedKVCache`（這支檔案）：只管「數值」，提供一塊大的實體記憶體
    （一個 tensor），並根據 BlockManager 給的 block table，把某個
    序列的 K/V 寫到正確的位置、或是從正確的位置讀出來。

物理上，整個池子是「一整塊」連續記憶體（跟 Stage 1 一樣，底層還是
一個 tensor，這點沒有變），差別在於：Stage 1 是「每個序列各自一塊」，
Stage 2 是「全部序列共用同一塊，用 block table 分配其中的小格子」。

    ┌───────────────────────────────────────────────────┐
    │ PagedKVCache 的實體池（k_pool，v_pool 結構相同）        │
    │                                                     │
    │ shape: [num_blocks, n_layers, n_heads, block_size, head_dim]
    │                                                     │
    │  block 0: [pos_in_block 0][1][2]...[block_size-1]   │
    │  block 1: [pos_in_block 0][1][2]...[block_size-1]   │
    │  ...                                                │
    │  block num_blocks-1: ...                            │
    │                                                     │
    │  ▲ 「block N 屬於哪個序列的第幾個邏輯 block」          │
    │     這件事完全不在這支檔案裡記錄，是 BlockManager 的責任 │
    └───────────────────────────────────────────────────┘

寫入/讀出時，都要把「序列內的絕對位置」換算成「(block table 裡的
第幾個 block, block 內的第幾格)」，這個換算就是 Paged 設計最核心的
一行數學：

    block table 裡的索引 = 絕對位置 // block_size
    block 內的偏移量     = 絕對位置 %  block_size
"""

from __future__ import annotations

import torch

from mini_vllm.models.tiny_transformer import TinyTransformerConfig


class PagedKVCache:
    def __init__(
        self,
        config: TinyTransformerConfig,
        num_blocks: int,
        block_size: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device

        pool_shape = (
            num_blocks,
            config.n_layers,
            config.n_heads,
            block_size,
            config.head_dim,
        )
        self.k_pool = torch.zeros(pool_shape, device=device)
        self.v_pool = torch.zeros(pool_shape, device=device)

    def write(
        self,
        layer_idx: int,
        start_pos: int,
        k: torch.Tensor,
        v: torch.Tensor,
        block_table: list[int],
    ) -> None:
        """
        把新算出來的 K/V 寫進 block 池裡，位置由 block_table 決定。

        k, v: [1, n_heads, T_new, head_dim]（Stage 2 範例維持 batch=1，
              多序列同時батch化留給 Stage 3 的 continuous batching）
        block_table: 這個序列目前的實體 block 編號列表（由
              BlockManager.ensure_capacity(...) 取得，呼叫前必須
              確保已經有足夠的 block）。

        實作上逐 token 寫入（而不是整批向量化），是刻意選擇：
        一批新 token 可能跨越多個 block 的邊界（例如 prefill 一個
        跨 3 個 block 的 prompt），逐 token 寫最直覺、最不容易寫錯，
        對於這個教學規模的模型，效能也完全足夠。等到 Stage 6
        討論效能優化時，才是用向量化/CUDA kernel 取代這段迴圈的時機。
        """
        T_new = k.shape[2]
        for i in range(T_new):
            pos = start_pos + i
            block_idx_in_table = pos // self.block_size
            offset = pos % self.block_size

            if block_idx_in_table >= len(block_table):
                raise IndexError(
                    f"位置 {pos} 需要第 {block_idx_in_table} 個邏輯 block，"
                    f"但 block_table 只有 {len(block_table)} 個 block。"
                    "呼叫 write() 前，必須先呼叫 "
                    "BlockManager.ensure_capacity() 確保 block 數量足夠。"
                )
            physical_block_id = block_table[block_idx_in_table]

            self.k_pool[physical_block_id, layer_idx, :, offset] = k[0, :, i]
            self.v_pool[physical_block_id, layer_idx, :, offset] = v[0, :, i]

    def read(
        self, layer_idx: int, end_pos: int, block_table: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        讀出某序列從位置 0 到 end_pos（不含）的所有 K/V，
        即使這些位置實際上散落在不連續的實體 block 裡。

        做法：先用 PyTorch 的 advanced indexing，把 block_table 指到的
        所有實體 block 一次「聚集（gather）」出來，再攤平成連續的
        位置維度，最後裁到剛好 end_pos 長度（最後一個用到的 block
        可能還有幾格是還沒寫入的殘留空間，要裁掉）。

        回傳: k, v 皆為 [1, n_heads, end_pos, head_dim]
        """
        num_blocks_needed = (end_pos + self.block_size - 1) // self.block_size
        used_block_ids = block_table[:num_blocks_needed]

        # 進階索引：一次把好幾個（可能不連續的）實體 block 取出來。
        # k_pool[used_block_ids, layer_idx] -> [num_blocks_needed, n_heads, block_size, head_dim]
        gathered_k = self.k_pool[used_block_ids, layer_idx]
        gathered_v = self.v_pool[used_block_ids, layer_idx]

        # 把「block 維度」跟「block 內位置維度」攤平合併成單一的
        # 「序列位置」維度： [num_blocks, n_heads, block_size, head_dim]
        #   -> permute -> [n_heads, num_blocks, block_size, head_dim]
        #   -> reshape -> [n_heads, num_blocks*block_size, head_dim]
        n_heads = self.config.n_heads
        head_dim = self.config.head_dim
        gathered_k = gathered_k.permute(1, 0, 2, 3).reshape(n_heads, -1, head_dim)
        gathered_v = gathered_v.permute(1, 0, 2, 3).reshape(n_heads, -1, head_dim)

        # 裁到剛好 end_pos（最後一個 block 可能還有沒用到的殘留空間）
        k = gathered_k[:, :end_pos, :].unsqueeze(0)  # [1, n_heads, end_pos, head_dim]
        v = gathered_v[:, :end_pos, :].unsqueeze(0)
        return k, v
