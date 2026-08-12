"""
Stage 1：支援 KV Cache 的 TinyTransformer。

跟 Stage 0 的 `mini_vllm/models/tiny_transformer.py` 幾乎一模一樣的
架構（一樣的 TinyTransformerConfig、一樣的 MLP、一樣的殘差連接
順序），只有 attention 換成 Stage 1 的 CausalSelfAttentionWithCache，
並且 forward 多了 `kv_cache` 跟 `start_pos` 兩個參數。

MLP 直接重用 Stage 0 的 `MLP` class —— 它本來就沒有跨 step 的狀態
需要快取（每個 token 的 MLP 輸出只跟它自己的向量有關，不需要看
其他 token），所以完全不需要為它另外寫快取版本。**這也是為什麼
KV cache 只叫「KV」cache，而不是「所有中間結果」cache**：只有
attention 需要「看到其他 token 的資訊」，而那份資訊就是 K 和 V。

呼叫方式（對照 Stage 0 的一次性 forward，Stage 1 拆成 prefill + decode 兩段）：

    cache = KVCache(config, batch_size=1)

    # prefill: 一次把整個 prompt 丟進去，start_pos=0
    logits = model(prompt_ids, kv_cache=cache, start_pos=0)
    cache.advance(prompt_ids.shape[1])

    # decode: 每次只丟「上一步生成的新 token」進去
    next_token = ...(從 logits 最後一個位置取樣)
    for _ in range(max_new_tokens - 1):
        logits = model(next_token_ids, kv_cache=cache, start_pos=cache.length)
        cache.advance(1)
        next_token = ...(從 logits 取樣)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mini_vllm.layers.attention import CausalSelfAttentionWithCache
from mini_vllm.layers.kv_cache import KVCache
from mini_vllm.models.tiny_transformer import MLP, TinyTransformerConfig


class TransformerBlockKV(nn.Module):
    """
    跟 Stage 0 的 TransformerBlock 結構完全相同
    （LayerNorm -> Attention -> 殘差 -> LayerNorm -> MLP -> 殘差），
    差別只在 attn 換成有 cache 的版本，forward 需要多傳
    kv_cache / layer_idx / start_pos 下去。
    """

    def __init__(self, config: TinyTransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.attn = CausalSelfAttentionWithCache(config)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, kv_cache: KVCache, start_pos: int) -> torch.Tensor:
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
        x = x + self.attn(self.ln1(x), kv_cache, self.layer_idx, start_pos)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformerKV(nn.Module):
    """
    Stage 1 版本的 TinyTransformer：架構跟 Stage 0 相同，
    但 forward 需要外部傳入 KVCache，並且用 start_pos 告知
    「這次傳進來的 token，在整個序列裡的絕對位置是從哪裡開始」。

    正確性驗證的標準做法（見 tests/test_stage1_kv_cache.py）：
    把 Stage 0 訓練好的權重（或同樣 random init 的權重）直接
    load_state_dict 進這個模型，然後：
      (a) 用「一次性 prefill 整個序列」的方式跑，logits 應該跟
          Stage 0 的一次性 forward **逐數值相等**（因為數學上是同一個
          計算，只是多了一層「先寫進 cache 再讀出來」的包裝）。
      (b) 用「prefill + 逐 token decode」的方式跑完整個生成流程，
          在貪婪取樣下，產生的文字應該跟 Stage 0 的 naive_generate
          **完全一致**。
    """

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlockKV(config, layer_idx=i) for i in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor, kv_cache: KVCache, start_pos: int
    ) -> torch.Tensor:
        """
        input_ids: [B, T]
          - prefill 時 T = prompt 長度
          - decode 時 T = 1
        start_pos: 這批 token 的絕對起始位置（見模組開頭的呼叫範例）

        回傳 logits: [B, T, vocab_size]

        注意：跟 Stage 0 最大的不同在這一行——位置編碼用的是
        「絕對位置」start_pos ~ start_pos+T-1，而不是每次都從 0 算起。
        這是必須的: decode 第 5 步生成的 token，它的位置編碼必須是
        「第 5 個位置」，而不是每次都被錯誤地當成「第 0 個位置」。

        被設計透過「進來的字數長度 (T)」以及「目前寫到第幾個字 (start_pos)」，
        來自動切換自己現在是處於 Prefill 還是 Decode 模式！

        ex.
        ```python
        # prefill: prompt 長度 10, T=10, start_pos=0
        # 外部把 prompt 塞 10 個字進來，起始位置為 0。
        # 模型就知道：「哦！我要一次幫這 10 個字加上位置標籤 0~9。」
        positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # decode 第 1 步: T=1, start_pos=10
        # 外部塞「剛剛生成的 1 個字，起始位置為 10」進來。
        # 模型就知道：「這只是第 10 個字，我只要幫這 1 個字貼上位置標籤 10 就好。」   
        positions = [10]

        # decode 第 2 步: T=1, start_pos=11
        # 外部塞「剛剛生成的 1 個字，起始位置為 11」進來。
        # 模型就知道：「這只是第 11 個字，我只要幫這 1 個字貼上位置標籤 11 就好。」
        positions = [11]
        ```
        """
        B, T = input_ids.shape
        end_pos = start_pos + T
        if end_pos > self.config.max_seq_len:
            raise ValueError(
                f"序列長度 {end_pos} 超過 max_seq_len {self.config.max_seq_len}"
            )

        # 這裡用「絕對位置」來建立位置編碼，並加到 token embedding 上。
        positions = torch.arange(start_pos, end_pos, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)  # [B, T, C]

        # 依序把每一層的 TransformerBlockKV 丟進去，注意每一層都要傳入
        # kv_cache / layer_idx / start_pos，讓它們可以正確地寫進快取、讀出快取、以及建立因果遮罩。
        for block in self.blocks:
            x = block(x, kv_cache, start_pos)

        # 這裡的 ln_f / lm_head 都是「每個 token 自己算自己的東西」，不需要跨 token 的資訊，所以不需要快取。
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]
        return logits
