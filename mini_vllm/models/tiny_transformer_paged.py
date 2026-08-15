"""
Stage 2：支援 PagedAttention 的 TinyTransformer。

架構跟 Stage 0 / Stage 1 完全相同（同一個 TinyTransformerConfig、
同一個 MLP），差別只在 attention 換成 PagedCausalSelfAttention，
forward 需要多傳一個 `block_table`。

呼叫方式（對照 Stage 1，多了 BlockManager 這一步）：

    block_manager = BlockManager(num_blocks=20, block_size=4)
    paged_cache = PagedKVCache(config, num_blocks=20, block_size=4)

    # prefill
    block_table = block_manager.ensure_capacity("seq-A", len(prompt_ids))
    logits = model(prompt_ids, paged_cache, start_pos=0, block_table=block_table)

    # decode（每一步都要先 ensure_capacity，可能觸發新 block 配置）
    seq_len = len(prompt_ids)
    for _ in range(max_new_tokens):
        block_table = block_manager.ensure_capacity("seq-A", seq_len + 1)
        logits = model(next_token_ids, paged_cache, start_pos=seq_len, block_table=block_table)
        seq_len += 1

    # 序列結束後記得歸還 block
    block_manager.free("seq-A")
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mini_vllm.layers.paged_attention import PagedCausalSelfAttention
from mini_vllm.layers.paged_kv_cache import PagedKVCache
from mini_vllm.models.tiny_transformer import MLP, TinyTransformerConfig


class TransformerBlockPaged(nn.Module):
    def __init__(self, config: TinyTransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.attn = PagedCausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        paged_cache: PagedKVCache,
        start_pos: int,
        block_table: list[int],
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), paged_cache, self.layer_idx, start_pos, block_table)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformerPaged(nn.Module):
    """
    Stage 2 版本的 TinyTransformer。跟 Stage 1 的 `TinyTransformerKV`
    唯一的差別，是 forward 多接受一個 `block_table` 參數，並把它
    一路往下傳給每一層的 attention。

    正確性驗證標準（見 tests/test_stage2_paged_attention.py）：
    把跟 Stage 0/1 完全相同的權重載入這個模型，跑完整個
    prefill + decode 生成流程，貪婪取樣下的輸出必須跟 Stage 0/1
    逐字元相同——PagedAttention 換的是「記憶體怎麼管理」，
    不應該改變 attention 算出來的任何數值。
    """

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlockPaged(config, layer_idx=i) for i in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        paged_cache: PagedKVCache,
        start_pos: int,
        block_table: list[int],
    ) -> torch.Tensor:
        B, T = input_ids.shape
        end_pos = start_pos + T
        if end_pos > self.config.max_seq_len:
            raise ValueError(
                f"序列長度 {end_pos} 超過 max_seq_len {self.config.max_seq_len}"
            )

        positions = torch.arange(start_pos, end_pos, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        for block in self.blocks:
            x = block(x, paged_cache, start_pos, block_table)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
