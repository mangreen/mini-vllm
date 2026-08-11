"""
Stage 0：迷你 decoder-only Transformer。

刻意不使用 torch.nn.TransformerEncoder/Decoder 這種高階封裝，
而是手刻每一層，理由：
  1. 之後的階段（KV cache、PagedAttention）都要深入修改 attention
     的內部實作，用高階封裝反而會擋路。
  2. Stage 0 的學習目標之一，就是把 attention 的每個矩陣運算形狀
     搞清楚——手刻一次比看十次文件有效。

架構（標準的 GPT 風格 decoder-only block）：

    input_ids [B, T]
        │
        ▼
    Token Embedding + Positional Embedding
        │
        ▼
    ┌─────────────────────────────┐
    │  TransformerBlock x N       │
    │  ┌───────────────────────┐  │
    │  │ LayerNorm             │  │
    │  │ Causal Self-Attention │  │
    │  │ + residual            │  │
    │  ├───────────────────────┤  │
    │  │ LayerNorm             │  │
    │  │ MLP (GELU)            │  │
    │  │ + residual            │  │
    │  └───────────────────────┘  │
    └─────────────────────────────┘
        │
        ▼
    Final LayerNorm
        │
        ▼
    LM Head (Linear -> vocab_size)
        │
        ▼
    logits [B, T, vocab_size]

Stage 0 版本每次 forward 都是「餵完整序列、重新算一次所有 token 的
attention」，這是刻意的：要先有一個正確、樸素的版本作為之後所有
優化（KV cache、PagedAttention...）的正確性對照組。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyTransformerConfig:
    vocab_size: int # 詞表大小
    hidden_dim: int = 128 # 隱藏層維度
    n_layers: int = 2 # Transformer block 層數
    n_heads: int = 4 # 注意力頭數
    max_seq_len: int = 256 # 最大序列長度
    mlp_ratio: int = 4 # MLP 隱藏層維度 = hidden_dim * mlp_ratio
    dropout: float = 0.0 # Dropout 機率

    def __post_init__(self) -> None:
        """
        hidden_dim 必須能被 n_heads 整除，才能平均分配到每個 head。

        ex. 若 hidden_dim=128, n_heads=4，則每個 head 的維度 = 128/4=32。
        """
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim({self.hidden_dim}) 必須能被 n_heads({self.n_heads}) 整除"
            )

    @property
    def head_dim(self) -> int:
        """
        每個注意力頭的維度 = hidden_dim / n_heads
        """
        return self.hidden_dim // self.n_heads


class CausalSelfAttention(nn.Module):
    """
    最樸素、沒有任何快取的 causal self-attention。
    簡單來說就是注意力機制模組。
    讓每個字去關注句子裡其他字之間的關係，但加上了「因果限制（Causal）」
    只能看當前與過去的字，絕對不能偷看未來的字。

    輸入向量 X ──> [ 矩陣投影 ] ──> Query (問題), Key (線索), Value (資訊)
                                   │
                                   ▼
                            算相似度 Q @ K^T
                                   │
                                   ▼
                   加上 Causal Mask (把未來的字蓋住)
                                   │
                                   ▼
                        Softmax 轉換機率 * Value

    Stage 0 每次呼叫都會把整個序列的 Q/K/V 重新算一次 —— 這正是
    之後階段要優化掉的地方（Stage 1 開始，K/V 會被快取起來，
    decode 時只需要算「新的那個 token」的 Q/K/V）。
    """

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads # 注意力頭數
        self.head_dim = config.head_dim # 每個注意力頭的維度

        self.qkv_proj = nn.Linear(config.hidden_dim, 3 * config.hidden_dim) # 投影成 Q/K/V
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim) # 輸出投影
        self.dropout = nn.Dropout(config.dropout) # Dropout 機率

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        注意力的真正運算過程。
        把資料拆成 Q, K, V 並切分成多個頭，計算注意力分數矩陣，
        接著疊加「下三角遮罩（Causal Mask）」，最後將計算結果組合輸出。

        ex. 假設 T=3，注意力分數矩陣 S = Q @ K^T，套用 causal mask 後：
        注意力分數矩陣 (T=3)           套用遮罩 (未來的字設為 -inf)
        [ S11  S12  S13 ]            [  S11  -inf  -inf ]  <-- 第1個字只能看自己
        [ S21  S22  S23 ]   ───>     [  S21   S22  -inf ]  <-- 第2個字能看 1, 2
        [ S31  S32  S33 ]            [  S31   S32   S33 ]  <-- 第3個字能看 1, 2, 3

        x: [B, T, hidden_dim]，回傳 [B, T, hidden_dim]
        
        ex. 
        假設 B=2, T=3, hidden_dim=4, n_heads=2，則：
        x: [2, 3, 4] -> qkv_proj -> [2, 3, 12] -> split -> q/k/v: [2, 3, 4] -> view -> [2, 3, 2, 2] -> transpose
        q/k/v: [2, 2, 3, 2] (B, n_heads, T, head_dim)
        attn_scores: [2, 2, 3, 3] (B, n_heads, T, T)
        attn_weights: [2, 2, 3, 3] (B, n_heads, T, T)
        out: [2, 2, 3, 2] -> transpose -> [2, 3, 4] (B, T, hidden_dim)

        注意：Stage 0 這裡永遠是「整個序列重新算一遍」，
        沒有任何跨呼叫的狀態被保留（沒有 KV cache）。
        """
        # x: [B, T, hidden_dim]
        B, T, C = x.shape

        # 投影成 Q/K/V，並拆成多頭
        qkv = self.qkv_proj(x)  # [B, T, 3*C]
        q, k, v = qkv.split(C, dim=-1)  # 各自 [B, T, C]

        # 拆成多頭： [B, T, n_heads, head_dim] -> [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # attention score: [B, n_heads, T, T]
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal mask：第 i 個 token 只能看到 <= i 的 token
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        # softmax + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 計算加權和，並把多頭結果組合回來
        out = attn_weights @ v  # [B, n_heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        return self.out_proj(out)


class MLP(nn.Module):
    """
    標準的 MLP（Feed Forward Network）模組。
    由兩層全連接層組成，中間夾著 GELU 激活函數，最後加上 Dropout。

    簡單來講就是多層感知機（前饋神經網路）。
    資料通過注意力層後，會送來這裡進行特徵的「放大 -> 非線性轉換 -> 縮回」，做二次深度加工。

    輸入 (hidden_dim: 128)
        │
        ▼  fc1 (放大 4 倍)
    中間層 (hidden * 4: 512) ──> GELU 激活函數 (引入非線性變換)
        │
        ▼  fc2 (壓回原本尺寸)
    輸出 (hidden_dim: 128)

    Stage 0 版本每次 forward 都是「餵完整序列、重新算一次所有 token 的 MLP」，這是刻意的：
    要先有一個正確、樸素的版本作為之後所有優化（KV cache、PagedAttention...）的正確性對照組。
    """
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim * config.mlp_ratio # MLP 隱藏層維度 = hidden_dim * mlp_ratio
        self.fc1 = nn.Linear(config.hidden_dim, hidden) # 第一層全連接層
        self.fc2 = nn.Linear(hidden, config.hidden_dim) # 第二層全連接層
        self.dropout = nn.Dropout(config.dropout) # Dropout 機率

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        執行特徵加工：x 先進入 fc1 放大維度，過 GELU 函數，再由 fc2 壓回原維度。  

        ex. 假設 B=2, T=3, hidden_dim=4, mlp_ratio=4，則：
        x: [2, 3, 4] -> fc1 -> [2, 3, 16] -> GELU -> [2, 3, 16] -> fc2 -> [2, 3, 4]
        """

        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    標準的 GPT 風格 decoder-only Transformer block。
    由兩個子模組組成：
    1. Causal Self-Attention + LayerNorm + 殘差連接
    2. MLP + LayerNorm + 殘差連接

        x (輸入)
          │
    ┌─────┴──────────────────┐
    │  LayerNorm (ln1)       │
    │  CausalSelfAttention   │
    └─────┬──────────────────┘
          ▼
        ( + ) <--- 殘差快線 1 (加上原本的 x)
          │
    ┌─────┴──────────────────┐
    │  LayerNorm (ln2)       │
    │  MLP                   │
    └─────┬──────────────────┘
          ▼
        ( + ) <--- 殘差快線 2
          │
        輸出
    """
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim) # LayerNorm 1
        self.attn = CausalSelfAttention(config) # Causal Self-Attention
        self.ln2 = nn.LayerNorm(config.hidden_dim) # LayerNorm 2
        self.mlp = MLP(config) # MLP

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        按照「正規化 -> 注意力 -> 殘差相加 -> 正規化 -> MLP -> 殘差相加」的順序計算並回傳。：
        1. LayerNorm -> Causal Self-Attention -> 殘差連接
        2. LayerNorm -> MLP -> 殘差連接

        ex.
        假設 B=2, T=3, hidden_dim=4，則：
        x: [2, 3, 4] -> ln1 -> [2, 3, 4] -> attn -> [2, 3, 4] -> ( + ) -> [2, 3, 4] -> 
        ln2 -> [2, 3, 4] -> mlp -> [2, 3, 4] -> ( + ) -> [2, 3, 4]
        """

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    """
    迷你版 GPT 風格 decoder-only Transformer。
    由多個 TransformerBlock 堆疊而成，最後接上 Final LayerNorm 與 LM Head (Linear -> vocab_size)。
    輸入 (input_ids: [B, T]) -> Token Embedding + Positional Embedding -> TransformerBlock x N -> Final LayerNorm -> LM Head -> logits: [B, T, vocab_size]
    
    input_ids [1, 3] (例如: [0, 1, 2])
        │
        ▼
    [ token_emb + pos_emb ] ──> 轉換為帶位置資訊的向量 [1, 3, 128]
        │
        ▼
    [ TransformerBlock 1 ] ──> 特徵萃取
        │
        ▼
    [ TransformerBlock 2 ] ──> 特徵萃取
        │
        ▼
    Final LayerNorm (ln_f)
        │
        ▼
    LM Head (Linear) ─────────> logits [1, 3, vocab_size] (每個位置預測下個字的得分)
    """
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config # TinyTransformerConfig

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim) # Token Embedding
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_dim) # Positional Embedding
        
        # Transformer block 堆疊
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_f = nn.LayerNorm(config.hidden_dim) # Final LayerNorm
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False) # LM Head (Linear -> vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T]，回傳 logits: [B, T, vocab_size]

        大腦思考全過程。
        輸入文字編號，加上位置訊息後，依序通過每一層 TransformerBlock，最後輸出每個位置預測下一個字的得分矩陣（logits）。

        ex. 
        假設 B=2, T=3, vocab_size=5，則：
        input_ids: [2, 3] (例如: [[0, 1, 2], [3, 4, 0]])
        token_emb: [2, 3, 128] (將每個 token 轉換為向量)
        pos_emb: [2, 3, 128] (將每個位置轉換為向量)
        x: [2, 3, 128] (token_emb + pos_emb)
        blocks: [2, 3, 128] (經過多層 TransformerBlock 特徵萃取)
        ln_f: [2, 3, 128] (Final LayerNorm)
        logits: [2, 3, 5] (每個位置預測下個字的得分，對應 vocab_size)

        注意：Stage 0 這裡永遠是「整個序列重新算一遍」，
        沒有任何跨呼叫的狀態被保留（沒有 KV cache）。
        """

        # 檢查輸入長度是否超過最大序列長度
        B, T = input_ids.shape
        if T > self.config.max_seq_len:
            raise ValueError(
                f"輸入長度 {T} 超過 max_seq_len {self.config.max_seq_len}"
            )

        # 位置編號 [0, 1, 2, ..., T-1]，並加上 batch 維度
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # [1, T]
        x = self.token_emb(input_ids) + self.pos_emb(positions)  # [B, T, C]

        # 依序通過每一層 TransformerBlock
        for block in self.blocks:
            x = block(x)

        # 最後的 LayerNorm 與 LM Head，輸出 logits
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]
        return logits
