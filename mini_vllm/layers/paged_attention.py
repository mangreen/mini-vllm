"""
Stage 2：PagedCausalSelfAttention —— 用 block table 讀寫 K/V 的 attention。

跟 Stage 1 的 `CausalSelfAttentionWithCache`（mini_vllm/layers/attention.py）
比較，數學完全一樣，唯一的差別在於：
  - Stage 1 的 KVCache 讀寫用「連續記憶體 + 絕對位置」直接 slice。
  - Stage 2 的 PagedKVCache 讀寫多了一層「透過 block_table 轉址」，
    因為這個序列的 K/V 可能散落在好幾個不連續的實體 block 裡。

這一層本身完全不需要知道「block 是怎麼分配的」——那是 BlockManager
的責任；它只需要拿到「這個序列目前的 block_table」，剩下的轉址、
gather 都交給 PagedKVCache 處理。這正是 PagedAttention 設計裡
「attention 計算」跟「記憶體管理」責任分離的精神。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_vllm.layers.paged_kv_cache import PagedKVCache
from mini_vllm.models.tiny_transformer import TinyTransformerConfig


class PagedCausalSelfAttention(nn.Module):
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.qkv_proj = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        paged_cache: PagedKVCache,
        layer_idx: int,
        start_pos: int,
        block_table: list[int],
    ) -> torch.Tensor:
        """
        x: [1, T, hidden_dim]（Stage 2 範例仍是單序列 batch=1，
           多序列一起跑會在 Stage 3 continuous batching 才處理）
        block_table: 這個序列目前持有的實體 block 編號列表

        跟 Stage 1 的 forward 逐行對照，會發現只有「寫進快取」跟
        「讀出快取」這兩步的呼叫方式不同（多傳了 block_table），
        其他數學（Q/K/V 投影、attention scores、因果遮罩、softmax）
        完全一模一樣——這也是為什麼正確性測試可以直接比較
        Stage 1 跟 Stage 2 的輸出是否相等。
        """
        B, T, C = x.shape
        assert B == 1, "Stage 2 範例維持 batch=1，多序列批次留給 Stage 3"

        qkv = self.qkv_proj(x)
        q, k_new, v_new = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [1, nh, T, hd]
        k_new = k_new.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v_new = v_new.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 跟 Stage 1 的差異只有這兩行：多傳一個 block_table，
        # 讓 PagedKVCache 知道要寫到/讀取全域池子的哪些格子。
        paged_cache.write(layer_idx, start_pos, k_new, v_new, block_table)
        end_pos = start_pos + T
        k_all, v_all = paged_cache.read(layer_idx, end_pos, block_table)

        attn_scores = (q @ k_all.transpose(-2, -1)) / math.sqrt(self.head_dim)

        q_pos = torch.arange(start_pos, end_pos, device=x.device)
        k_pos = torch.arange(0, end_pos, device=x.device)
        causal_mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v_all
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
