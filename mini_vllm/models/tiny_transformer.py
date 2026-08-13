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

---
#專有名詞說明:

## 1. Token Embedding（詞向量）：把文字變成數字密碼
電腦只認得 0 和 1，不認得「愛」這個字。透過 Token Embedding，把「愛」這個字轉換成一個向量（數字密碼），
ex. 「愛」 -> [0.8, 0.1, -0.5, 0.3, ...]，這個向量的維度就是 hidden_dim
------------------------------
超大詞彙表 (Embedding Matrix)
[我]: 0.2
[愛]: 0.8
[AI]: -0.1
[你]: 0.1
[他]: 0.3
------------------------------
[我][愛][AI] --> Token Embedding: [0.2][0.8][-0.5]
「我」的 Token 向量，在數學空間裡會跟「你」「他」的 Token 向量離得很近，這個距離就是「語意相似度」。

---
## 2. Positional Embedding（位置向量）：把字在句子裡的位置變成數字密碼
Transformer 模型本身沒有「順序」的概念，所以需要 Positional Embedding 來告訴模型「這個字在句子裡的位置」。
如果沒有位置，AI 會覺得「我打你」跟「你打我」是一模一樣的句子（因為都有「我、打、你」三個字）。
ex. 第 1 個字 -> [0.1, 0.2, 0.3, ...]，第 2 個字 -> [0.4, 0.5, 0.6, ...]，這個向量的維度也是 hidden_dim

---
## 3. Causal Self-Attention（因果自注意力）：
讓每個字去關注句子裡其他字之間的關係，但加上了「因果限制（Causal）」，只能看當前與過去的字，絕對不能偷看未來的字。
同一個字會先被投影成三個向量：
Q（輸入問題）、K（標籤）、V（輸出資訊）
ex. 輸入「我買了一台蘋果」
「蘋果」可能是「水果」也可能是「手機」，AI 會先問自己「蘋果是水果還是手機？」（Q），
再去找線索（K）：「我買了一台蘋果」這句話裡有「買」這個字，這個字的向量會告訴 AI「蘋果是水果」的線索，
最後 AI 會把「蘋果是 iPhone」這個資訊（V）輸出給下一層。

---
## 4. MLP（多層感知機）：把注意力層的輸出再加工一次，做二次特徵萃取。
MLP 由兩層全連接層組成，中間夾著 GELU 激活函數，最後加上 Dropout。
GELU，這個函數可以讓模型學到更複雜的非線性關係，比 ReLU 更平滑。
GELU 的核心哲學：「你越強，我越信任你；你越弱，我越可能放棄你。」

ex. 「我買了一台蘋果」 
-> MLP (GELU) 放大「Apple=蘋果」的特徵，縮小「水果=蘋果」的特徵，讓模型更容易學到「蘋果=Apple」這個關係 
->「好貴的 iPhone」

---
## 5. LayerNorm（層正規化）：把每個字的向量做正規化，讓數值分布更穩定，避免梯度爆炸或消失。
LayerNorm 的核心精神是：「每個人自己跟自己比」。
ex. 假設有 3 個字（同學），每個字的向量維度是 4（考了四科）：
字1（天才）: [97, 98, 99, 100] -> [0.1, 0.2, 0.3, 0.4]
字2（普通人）: [50, 60, 70, 80] ->[0.5, 0.6, 0.7, 0.8]
字3（白痴）: [10, 20, 30, 40] -> [0.9, 1.0, 1.1, 1.2]
LayerNorm 會對每個字（同學）的向量做正規化，讓每個字（同學）的向量分布更穩定

## 6. BatchNorm（批次正規化）：把整個 batch 的向量做正規化，讓數值分布更穩定，避免梯度爆炸或消失。
ex. 假設有 3 個字（同學），每個字的向量維度是 4（考了四科）：
字1（天才）: [97, 98, 99, 100] -> [-1.22, -1.22, -1.22, -1.22]
字2（普通人）: [50, 60, 70, 80] -> [0.00, 0.00, 0.00, 0.00]
字3（白痴）: [10, 20, 30, 40] -> [1.22, 1.22, 1.22, 1.22]

LayerNorm 與 BatchNorm 的差別：
- LayerNorm 是「每個人自己跟自己比」，BatchNorm 是「大家同部位一起比」。
- LayerNorm 適合 NLP，BatchNorm 適合 CV。

---
## 7. 殘差連接（Residual Connection）：把注意力層或 MLP 的輸出加回原本的輸入，讓梯度可以直接傳回去，避免梯度消失。
ex. 假設有一個字的向量是 [0.1, 0.2, 0.3, 0.4]，經過注意力層後變成 [0.5, 0.6, 0.7, 0.8]，殘差連接會把兩個向量相加，
[0.1, 0.2, 0.3, 0.4] + [0.5, 0.6, 0.7, 0.8] = [0.6, 0.8, 1.0, 1 .2]，這樣梯度就可以直接傳回去原本的向量，避免梯度消失。

ex. ❌ 沒加殘差時：
「蘋果」
-> 第 1 層：「Apple」
-> 第 2 層：「3C 產品」
...
-> 第 100 層：「一個會發光的長方形物體」

ex. ✅ 有加殘差時：
「蘋果」
-> 第 1 層：「蘋果+Apple」
-> 第 2 層：「蘋果+3C 產品」
...
-> 第 100 層：「蘋果+iPhone」

---
## 8. LM Head（語言模型頭）：把最後的向量轉換成每個字的得分，這個得分可以用來預測下一個字。
- Vocab Size (詞表大小)： 這台 AI 認得的所有字詞總數（通常是 3 萬到 10 萬個字，就像一本大字典）。
- logits（對數機率）：LM Head 的輸出就是 logits，這個向量的每個元素就是每個字的得分，得分越高的字，AI 越有可能選擇它作為下一個字。

ex. 假設 vocab_size=5，最後的向量是 [0.1, 0.2, 0.3, 0.4]，經過 LM Head 後變成 [0.5, 0.6, 0.7, 0.8, 0.9]，
這個向量的每個元素就是每個字的得分，得分越高的字，AI 越有可能選擇它作為下一個字。

像是拿著「蘋果」對著字典裡的每一個字一個一個問下一個字是你的機率有多高？
------------------------------
西瓜: -0.5
蘋果: 0.1
好貴: 9.3
iPhone: 8.4
------------------------------
最後，字典裡的幾萬個字都會得到一個原始分數（在 AI 裡叫做 Logits）。這個時候的分數有正有負，但還不是機率。

---
## 9. Softmax（機率轉換）：把 logits 轉換成機率，這個機率可以用來預測下一個字。
ex. 假設 logits=[-0.5, 0.1, 8.3, 9.4]，經過 Softmax 後變成 [0.0001, 0.0002, 0.2689, 0.7318]，這個向量的每個元素就是每個字的機率，機率越高的字，AI 越有可能選擇它作為下一個字。

「蘋果」的 Logits 分數經過 Softmax 後，變成機率分布：
------------------------------
西瓜: 0.01%
蘋果: 0.02%
好貴: 83%
iPhone: 26%
------------------------------
「我買了一台蘋果」 ➔ AI 成功吐出：『好貴』！

---
# 📊 總結 Transformer 的一生

| 階段 | 負責組件 | 白話在幹嘛？
| --- | --- | --- |
| 輸入 | Embedding | 把「我買了一台蘋果」轉成一堆原始的數字向量（剛入學的學生）。
| 修行 | Self-Attention | 讓字跟字交流，讓「蘋果」和「一台」對上眼神，確定是 3C 產品。
| 修煉 | MLP (GELU) | 每個字閉關，把 3C 的概念昇華成「科技、高薪、信仰」等高維特徵。
| 保命 | Residual + Norm | 確保學生修行時不會走火入魔（退化），適時保留初心、自己跟自己比。
| 發言 | LM Head (Linear) | 拿著大腦特徵去對照整本字典，算出字典裡每個字的得分。
| 輸出 | Softmax | 把得分變機率，大聲噴出機率最高的下一個字：「好貴」！

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

        # 💥 重點 1：對「長度為 T 的所有 token」重新投影出 Q, K, V 矩陣
        # 投影成 Q/K/V，並拆成多頭
        # ex. 假設 B=2, T=3, hidden_dim=4, n_heads=2，則：
        # qkv_proj: [2, 3, 12] (B, T, 3*hidden_dim)
        # split: q/k/v: [2, 3, 4] (B, T, hidden_dim)
        # 拆成多頭: [2, 3, 4] -> view -> [2, 3, 2, 2] (B, T, n_heads, head_dim) -> transpose -> [2, 2, 3, 2] (B, n_heads, T, head_dim)
        qkv = self.qkv_proj(x)  # [B, T, 3*C]
        q, k, v = qkv.split(C, dim=-1)  # 各自 [B, T, C]

        # 拆成多頭： [B, T, n_heads, head_dim] -> [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 💥 重點 2：計算 T x T 的所有注意力分數 (q @ k.T)
        # attention score: [B, n_heads, T, T]
        # 這裡的 attn_scores 是每個 token 對其他 token 的注意力分數矩陣。
        # 這裡會算第 1 個字對第 1 個字、第 2 個字對第 1~2 個字... 全部重算！
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

        # 💥 這裡：不管舊 token 算過幾次，全部重新算 Embedding！
        # 位置編號 [0, 1, 2, ..., T-1]，並加上 batch 維度
        # ex. T=3 -> positions: [0, 1, 2] -> unsqueeze(0) -> [1, 3]
        # 這裡使用 unsqueeze(0) 是為了讓 positions 的形狀與 input_ids 對齊，方便後續的加法運算。
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # [1, T]
        x = self.token_emb(input_ids) + self.pos_emb(positions)  # [B, T, C]

        # 依序通過每一層 TransformerBlock
        for block in self.blocks:
            x = block(x)

        # 最後的 LayerNorm 與 LM Head，輸出 logits
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]
        return logits
