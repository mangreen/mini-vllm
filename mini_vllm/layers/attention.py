"""
Stage 1：支援 KV Cache 的 causal self-attention。

跟 Stage 0 的 CausalSelfAttention（mini_vllm/models/tiny_transformer.py）
比較，差異只有三個地方，其他數學完全一樣：

  1. Q/K/V 投影只對「這次傳進來的 x」做（prefill 是整個 prompt，
     decode 是 1 個新 token），不會對已經在快取裡的舊 token 重算。
  2. 算出新的 K/V 之後，先寫進 KVCache，再把「快取裡全部（含新寫入）
     的 K/V」讀出來用於 attention —— 這一步是舊版本沒有的。
  3. causal mask 要用「絕對位置」來判斷，而不是單純看 T x T 的
     下三角矩陣，因為 query 的位置是 start_pos ~ start_pos+T-1，
     但 key 的位置是 0 ~ start_pos+T-1（含所有快取的舊 token）。

    Stage 0（無 cache）：                Stage 1（有 cache）：
    ┌─────────────────────┐             ┌─────────────────────────────┐
    │ 每次都用「完整序列」   │             │ 只用「這次新進來的 x」算 Q/K/V  │
    │ 重新算 Q/K/V         │             │ 舊 K/V 直接從 cache 讀出來    │
    │                     │             │                             │
    │ attn_scores: [T, T] │             │ attn_scores: [T, start+T]   │
    └─────────────────────┘             └─────────────────────────────┘

因果遮罩（causal mask）的通用公式：
  給定 query 的絕對位置範圍 [start_pos, start_pos+T)，
  key 的絕對位置範圍 [0, start_pos+T)，
  第 i 個 query（絕對位置 start_pos+i）只能看到
  絕對位置 <= start_pos+i 的 key。

這個公式在兩種情況下都對：
  - prefill（start_pos=0, T=prompt_len）：
    退化成跟 Stage 0 一模一樣的 T x T 下三角遮罩。
  - decode（T=1, start_pos=已快取長度）：
    這個唯一的 query 可以看到「所有」已快取的 key（因為它們
    全部都在它之前），完全不需要遮罩。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_vllm.layers.kv_cache import KVCache
from mini_vllm.models.tiny_transformer import TinyTransformerConfig


class CausalSelfAttentionWithCache(nn.Module):
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
        kv_cache: KVCache,
        layer_idx: int,
        start_pos: int,
    ) -> torch.Tensor:
        """
        x: [B, T, hidden_dim]
          - prefill 時 T = prompt 長度
          - decode 時 T = 1（每次只丟「新生成的那一個 token」進來）
        kv_cache: 這一次 forward 共用的 KV 快取物件
        layer_idx: 目前是第幾層（快取要寫到快取的哪一層）
        start_pos: 這批 token 在整個序列裡的起始絕對位置
          - prefill 時 start_pos = 0
          - decode 第一步時 start_pos = prompt 長度
          - decode 第二步時 start_pos = prompt 長度 + 1，以此類推

        回傳: [B, T, hidden_dim]
        """

        # 步驟 0：拆出 B, T, C
        B, T, C = x.shape

        # 步驟 1：只對「這次新進來的 x」投影 Q/K/V，
        #        不是整個序列 —— 這是跟 Stage 0 最根本的差異。
        qkv = self.qkv_proj(x)  # [B, T, 3*C]
        q, k_new, v_new = qkv.split(C, dim=-1)  # 各自 [B, T, C]

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, nh, T, hd]
        k_new = k_new.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v_new = v_new.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 步驟 2：把新算出來的 K/V 寫進快取，然後讀出「從頭到現在」的全部 K/V。
        kv_cache.write(layer_idx, start_pos, k_new, v_new)
        end_pos = start_pos + T
        k_all, v_all = kv_cache.read(layer_idx, end_pos)  # [B, nh, end_pos, hd]

        # 步驟 3：attention scores，形狀是 [B, nh, T, end_pos]
        #        —— 注意這裡「不是」T x T，而是 T x end_pos，
        #        因為 query 只有這次新進來的 T 個，但 key 是全部快取的 end_pos 個。
        attn_scores = (q @ k_all.transpose(-2, -1)) / math.sqrt(self.head_dim)

        """
        步驟 4：用「絕對位置」建立 💥 關鍵因果遮罩(Causal Mask)。
        q_pos: 這次 T 個 query 的絕對位置，例如 start_pos=5, T=3 -> [5, 6, 7]
        k_pos: 目前快取裡全部 key 的絕對位置，例如 [0, 1, ..., end_pos-1]
        mask[i, j] = True 代表「第 i 個 query 看不到第 j 個 key」，
        也就是 j（key 的絕對位置）比 i（query 的絕對位置）還晚。

        ex.
        # prefill: start_pos=0, T=3, end_pos=3
        # 跟 Stage 0 一模一樣，產生一個 3 x 3 的 causal mask，因為前面的字不能偷看後面的字。
        q_pos = [0, 1, 2]
        k_pos = [0, 1, 2]
        causal_mask = [[False, True,  True ],
                       [False, False, True ],
                       [False, False, False]]

        # decode 第 1 步: start_pos=3, T=1, end_pos=4
        # 產生一個 1 x 4 的 causal mask，唯一的 query 可以看到所有已快取的 key。
        q_pos = [3]
        k_pos = [0, 1, 2, 3]
        causal_mask = [[False, False, False, False]]

        # decode 第 2 步: start_pos=4, T=1, end_pos=5
        # 產生一個 1 x 5 的 causal mask，唯一的 query 可以看到所有已快取的 key。
        q_pos = [4]
        k_pos = [0, 1, 2, 3, 4]
        causal_mask = [[False, False, False, False, False]]
        """
        q_pos = torch.arange(start_pos, end_pos, device=x.device)  # [T]
        k_pos = torch.arange(0, end_pos, device=x.device)  # [end_pos]
        causal_mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)  # [T, end_pos]
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf")) # [B, nh, T, end_pos]

        # 步驟 5：softmax + dropout + matmul，得到最後的輸出。
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 步驟 6：把注意力權重乘上全部的 V，得到最後的輸出。
        out = attn_weights @ v_all  # [B, nh, T, hd]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        return self.out_proj(out)
