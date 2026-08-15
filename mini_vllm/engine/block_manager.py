"""
Stage 2：PagedAttention 的核心 —— BlockManager。

Stage 1 的 KVCache 有一個根本缺陷：**每個序列一開口，就要預先配置
一整塊 `[max_seq_len, ...]` 大小的連續記憶體**，不管這個序列實際上
會用到多少。100 個並發序列、每個都預留 max_seq_len，記憶體很快就
爆了；而且序列之間的記憶體完全獨立，即使兩個序列有一大段內容一模
一樣（例如共用的 system prompt），也沒辦法共享。

PagedAttention 借用作業系統「分頁記憶體（paging）」的概念解決這個
問題：

    作業系統的虛擬記憶體                PagedAttention 的 KV cache
    ─────────────────────              ─────────────────────────
    行程的虛擬位址空間                    序列的「邏輯」token 位置
    固定大小的 page（頁）                固定大小的 block（存 N 個 token 的 K/V）
    page table（虛擬頁 -> 實體頁）        block table（邏輯 block -> 實體 block）
    實體記憶體（不連續也沒關係）           一個全域共用的實體 KV cache 池

關鍵想法：**不要求一個序列的 K/V 存在連續記憶體裡**。而是把整個
KV cache 切成很多固定大小的「block」，序列需要多少就跟全域的
free pool 要幾個 block，記憶體不夠時再要，用完了就還回去。
一個序列的「block table」，就是一份記錄「我的第 i 個 block，
實際上是全域池子裡的第幾個 block」的對照表——這也是「Paged」這個
名字的由來。

這支檔案只負責「block 的配置/釋放/記帳」這個邏輯層，**不涉及任何
實際的 K/V 數值**——數值的讀寫是 `mini_vllm/layers/paged_kv_cache.py`
的工作。這是刻意的責任分離：BlockManager 只回答「這個序列可以用
哪些 block」，不管「block 裡面存的是什麼」。
"""

from __future__ import annotations

from collections import deque


class BlockManager:
    """
    管理一個「全域共用」的 block 池，以及每個序列各自的 block table。

    ex. block_size=4 的情況下，一個長度 10 的序列需要
    ceil(10/4) = 3 個 block（最後一個 block 只用了 2 格，其他 2 格空著）：

        序列的邏輯位置：  [0 1 2 3][4 5 6 7][8 9 _ _]
                          block 0   block 1   block 2（邏輯編號）
                             │         │         │
                             ▼         ▼         ▼
        block table：      [ 7,        2,        5 ]   <- 實體 block 編號（可以不連續！）

    這裡 block table = [7, 2, 5] 的意思是：這個序列的邏輯 block 0，
    實際存在全域池子的第 7 號 block；邏輯 block 1 存在第 2 號；
    邏輯 block 2 存在第 5 號。這三個實體 block 在記憶體裡完全不需要
    相鄰——這正是「Paged」的精神：不要求連續，只要記得對照表。
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size

        # free_block_ids：目前還沒被任何序列使用的實體 block 編號。
        # 用 deque 當作一個簡單的 pool：需要 block 時從左邊拿，
        # 釋放時放回右邊（先進先出，方便觀察 block 被重複使用的行為）。
        self.free_block_ids: deque[int] = deque(range(num_blocks)) # 放入 0 到 num_blocks-1

        # block_tables：seq_id -> 這個序列目前持有的實體 block 編號列表
        # （順序就是這個序列的邏輯 block 順序，見上面的圖）。
        self.block_tables: dict[str, list[int]] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def num_blocks_needed(self, num_tokens: int) -> int:
        """存下 num_tokens 個 token 的 K/V，總共需要幾個 block。"""
        return (num_tokens + self.block_size - 1) // self.block_size  # 無條件進位

    def ensure_capacity(self, seq_id: str, num_tokens: int) -> list[int]:
        """
        確保某序列有足夠的 block 能存下 num_tokens 個 token。

        這個方法是**冪等（idempotent）**且**只增不減**的：如果序列
        現有的 block 已經夠用，什麼都不做；如果不夠，只補上差額。
        這樣同一個方法可以同時處理兩種情境：

          - **prefill**：`ensure_capacity(seq_id, prompt_len)`，
            一次配置好整個 prompt 需要的 block。
          - **decode**：每生成 1 個新 token 後呼叫
            `ensure_capacity(seq_id, cache.length + 1)`，通常什麼都
            不用做（上一個 block 還有空位），只有在剛好跨過 block
            邊界時，才會真的去 free pool 拿 1 個新 block。

        回傳：這個序列目前的完整 block table（實體 block 編號列表）。
        """
        needed = self.num_blocks_needed(num_tokens)
        table = self.block_tables.setdefault(seq_id, [])

        while len(table) < needed:
            if not self.free_block_ids: # 當 free_block_ids 長度小於或等於 0 時
                raise MemoryError(
                    f"KV cache block 池已滿（共 {self.num_blocks} 個 block，"
                    f"block_size={self.block_size}），序列 {seq_id!r} "
                    f"無法再配置新 block。這就是 PagedAttention 裡"
                    "「記憶體不夠時要嘛拒絕新請求、要嘛搶佔（preempt）"
                    "某個序列」這個排程問題的來源，Stage 3 會處理。"
                )
            table.append(self.free_block_ids.popleft())

        return table

    def get_block_table(self, seq_id: str) -> list[int]:
        return self.block_tables.get(seq_id, [])

    def free(self, seq_id: str) -> None:
        """釋放某序列持有的所有 block，放回全域 free pool 給其他序列用。"""
        table = self.block_tables.pop(seq_id, [])
        self.free_block_ids.extend(table)

    def memory_usage_summary(self) -> dict[str, int]:
        """回傳目前 block 池的使用狀況，方便觀察/測試/示範記憶體效率。"""
        used = self.num_blocks - self.num_free_blocks
        return {
            "total_blocks": self.num_blocks,
            "used_blocks": used,
            "free_blocks": self.num_free_blocks,
            "num_sequences": len(self.block_tables),
        }
