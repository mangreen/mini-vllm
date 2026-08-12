"""
Stage 1：手刻 KV Cache。

Stage 0 的 CausalSelfAttention 每次呼叫，都要把「目前為止的整個序列」
重新投影出 Q/K/V、重新算一次 attention。但其實：對於「舊」的 token，
它們的 K、V 向量在每一步都是完全相同的數值（只跟 token 本身的內容、
它所在的位置有關，跟後面又生成了什麼新 token 無關）。

KV Cache 的想法很單純：既然 K、V 算過就不會變，那就把它們存起來，
之後每個 decode step 只需要：
  1. 算「新的那一個 token」的 Q、K、V（而不是整個序列）
  2. 把新的 K、V 寫進快取
  3. 用新 token 的 Q，去跟「快取裡全部（含新寫入的）K、V」做 attention

這支檔案定義的 KVCache，就是那個「快取」本身：一塊預先配置好的
連續記憶體（tensor），依照 [batch, layer, position, head, head_dim]
的順序把每一層、每個位置的 K/V 存起來。

    ┌──────────────────────────────────────────────────────┐
    │ KVCache (以 k_cache 為例，v_cache 結構相同）            │
    │                                                      │
    │ shape: [B, n_layers, n_heads, max_seq_len, head_dim] │
    │                                                      │
    │  layer 0: [pos0][pos1][pos2]...[pos N-1]             │
    │  layer 1: [pos0][pos1][pos2]...[pos N-1]             │
    │  ...                                                 │
    │                                                      │
    │  ▲ prefill 時一次寫入 pos 0 ~ prompt_len-1             │
    │  ▲ 之後每個 decode step 只多寫 1 個新位置                │
    └──────────────────────────────────────────────────────┘

這是 Stage 1 的版本：**每個序列各自獨立配置一整塊 [max_seq_len, ...]
大小的記憶體**（即使序列實際長度遠小於 max_seq_len，還是預先整塊配
好）。這樣做的缺點——記憶體浪費、無法跨序列共享——正是 Stage 2
PagedAttention 要解決的問題。Stage 1 先求正確、求懂，不求省記憶體。
"""

from __future__ import annotations

import torch

from mini_vllm.models.tiny_transformer import TinyTransformerConfig


class KVCache:
    """
    管理「一個 batch 的所有序列、所有層」的 K/V 快取。

    使用方式（典型的一次 prefill + 多次 decode）：

        cache = KVCache(config, batch_size=1)

        # prefill：一次寫入整個 prompt 的 K/V
        for layer_idx, block in enumerate(model.blocks):
            k, v = ...(這層算出來的 K, V，形狀 [B, n_heads, prompt_len, head_dim])
            cache.write(layer_idx, start_pos=0, k=k, v=v)

        # decode：每個 step 只寫入 1 個新位置
        for layer_idx, block in enumerate(model.blocks):
            k_new, v_new = ...(這層算出來的 K, V，形狀 [B, n_heads, 1, head_dim])
            cache.write(layer_idx, start_pos=cache.length, k=k_new, v=v_new)
        cache.advance(1)
    """

    def __init__(
        self,
        config: TinyTransformerConfig,
        batch_size: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.batch_size = batch_size
        self.device = device

        # 預先配置整塊記憶體，全部先填 0。
        # shape: [B, n_layers, n_heads, max_seq_len, head_dim]
        #
        # 注意順序：n_heads 放在 max_seq_len 前面，是為了跟
        # attention.py 裡拆完多頭之後的 K/V 形狀 [B, n_heads, T, head_dim]
        # 直接對齊，寫入/讀出時才不需要額外 transpose。
        cache_shape = (
            batch_size,
            config.n_layers,
            config.n_heads,
            config.max_seq_len,
            config.head_dim,
        )

        """
        這裡用兩塊記憶體分開存 K/V，因為它們的數值完全不同，
        也不會互相用到對方的數值，分開存可以避免不必要的 memory access，對效能有幫助。
        另外，這裡用 torch.zeros() 而不是 torch.empty()，是為了避免 debug 時出現「cache 裡有奇怪的垃圾值」的情況
        
        ex.
        ```python
        torch.empty(2, 3) # 创建一个 2x3 的未初始化张量，里面可能包含随机的垃圾值
        # tensor([[1.4013e-45,  0.0000e+00,  0.0000e+00],
        #         [0.0000e+00,  0.0000e+00,  0.0000e+00]])
        
        torch.zeros(2, 3) # 创建一个 2x3 的全 0 张量
        # tensor([[0., 0., 0.],
        #         [0., 0., 0.]])
        ```
        """
        self.k_cache = torch.zeros(cache_shape, device=device)
        self.v_cache = torch.zeros(cache_shape, device=device)

        # length：目前快取裡「有效」的位置數（所有層共用同一個長度，
        # 因為同一個 step 裡，每一層都會被寫入同樣多的新 token）。
        self.length = 0

    def write(self, layer_idx: int, start_pos: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        把某一層算出來的新 K/V，寫進快取的指定位置。

        k, v: [B, n_heads, T_new, head_dim]
          - prefill 時 T_new = prompt 長度，start_pos = 0
          - decode 時 T_new = 1，start_pos = 目前已快取的長度

        注意：這裡「不會」自動移動 self.length，因為一個 step 裡
        會對每一層都呼叫一次 write，必須全部層都寫完之後，
        才由呼叫端統一呼叫 advance() 移動一次長度指標。
        """
        T_new = k.shape[2]
        end_pos = start_pos + T_new
        if end_pos > self.config.max_seq_len:
            raise ValueError(
                f"KV cache 已滿：嘗試寫到位置 {end_pos}，"
                f"但 max_seq_len 只有 {self.config.max_seq_len}。"
                "（這正是 Stage 1 的已知限制，Stage 2 PagedAttention "
                "會用動態配置 block 的方式解決固定上限的問題）"
            )

        """
        寫入快取的方式很簡單：
        1. 先算出要寫入的範圍 end_pos = start_pos + T_new
        2. 用切片把快取的對應位置（k_cache[:, layer_idx, :, start_pos:end_pos]）替換成新的 K/V

        ex.
        k_cache[:, 0, :, 0:2] = k  (將 k 寫入 Layer 0 的前兩格)

        Layer 0 (更新後):
        Head 0:  [ [k1],  [k2],  0.0,  0.0 ] <- 覆蓋了前兩格
        Head 1:  [ [k3],  [k4],  0.0,  0.0 ] <- 覆蓋了前兩格
        Layer 1 (保持不變):
        Head 0:  [ 0.0,   0.0,  0.0,  0.0 ]
        Head 1:  [ 0.0,   0.0,  0.0,  0.0 ]

        k_cache[:, 0, :, 2:3] = k  (將新 k 寫入 Layer 0 的第三格)

        Layer 0 (更新後):
        Head 0:  [ [k1],  [k2],  [新k],  0.0 ] <- 精準寫入第三格
        Head 1:  [ [k3],  [k4],  [新k],  0.0 ] <- 精準寫入第三格
        Layer 1 (保持不變):
        Head 0:  [ 0.0,   0.0,   0.0,  0.0 ]
        Head 1:  [ 0.0,   0.0,   0.0,  0.0 ]
        """
        self.k_cache[:, layer_idx, :, start_pos:end_pos] = k
        self.v_cache[:, layer_idx, :, start_pos:end_pos] = v

    def read(self, layer_idx: int, end_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        讀出某一層、從位置 0 到 end_pos（不含）的所有 K/V。

        回傳: k, v 皆為 [B, n_heads, end_pos, head_dim]

        ex.
        # k_cache
        Layer 0:
        Head 0:  [ [k1],  [k2],  [k3],  [k4] ]
        Head 1:  [ [k5],  [k6],  [k7],  [k8] ]
        Layer 1:
        Head 0:  [ [k9],  [k10], [k11], [k12] ]
        Head 1:  [ [k13], [k14], [k15], [k16] ]

        # v_cache
        Layer 0:
        Head 0:  [ [v1],  [v2],  [v3],  [v4] ]
        Head 1:  [ [v5],  [v6],  [v7],  [v8] ]
        Layer 1:
        Head 0:  [ [v9],  [v10], [v11], [v12] ]
        Head 1:  [ [v13], [v14], [v15], [v16] ]

        read(layer_idx=0, end_pos=2) 會回傳：
        k:
        Head 0:  [ [k1],  [k2] ]
        Head 1:  [ [k5],  [k6] ]
        v:
        Head 0:  [ [v1],  [v2] ]
        Head 1:  [ [v5],  [v6] ]
        """
        k = self.k_cache[:, layer_idx, :, :end_pos]
        v = self.v_cache[:, layer_idx, :, :end_pos]
        return k, v

    def advance(self, num_new_tokens: int) -> None:
        """一個 step（所有層都寫完之後）呼叫一次，移動長度指標。"""
        self.length += num_new_tokens

    def reset(self) -> None:
        """清空快取，長度歸零（重新開始一個新序列時用）。"""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.length = 0
