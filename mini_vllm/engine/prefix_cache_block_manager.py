"""
Stage 4：PrefixCachingBlockManager —— 讓內容一模一樣的 block 可以被
多個序列共用，不需要重複配置、更不需要重複計算 K/V。

這支檔案繼承自 Stage 2 的 `BlockManager`，加上兩個新概念：

  1. **內容定址（content-addressed）**：每個「裝滿了」的 block，
     用它裝的 token 內容算一個 hash，登記進一份全域索引
     （`hash_to_block`）。之後任何序列，只要前綴的某一段 token
     跟這個 hash 對得上，就能直接「借用」這個 block——不用重新
     配置記憶體，**更不用重新跑一次 forward 把 K/V 算出來**，
     這是 Stage 2 單純的記憶體管理省不到的效能。

  2. **參照計數（reference counting）**：一個 block 可能同時被好幾個
     序列用（`ref_counts[block_id]`），只有當所有用它的序列都不再
     需要它了（ref_count 歸零），才真的把它放回 free pool。而且
     放回 free pool 之後，它的內容 hash 索引**依然保留**——只要
     還沒被別的新內容蓋掉，之後有序列剛好又需要一模一樣的前綴，
     還是可以立刻「復活」它，不用重新計算。

## 為什麼這個設計不需要 copy-on-write

Prefix caching 教學文章常常會強調 copy-on-write（COW）：如果一個
被共用的 block 之後需要被修改，要先複製一份出來，不能直接改，
否則會弄髒其他序列的資料。

這裡刻意用一個設計上的簡化，讓 COW 變得**不需要**：**每個序列的
prompt，它的最後一個邏輯 block，永遠是這個序列私有、新配置的
block，不會拿去跟快取比對，也不會被登記成可共用的項目**——即使
它的內容剛好跟別人一模一樣。

原因：block 快取的是「K/V 數值」，不是「logits」。要生成 prompt
之後的第一個新 token，一定得針對 prompt 最後一個位置**真的跑一次
forward** 拿到 logits，而 forward 的過程一定會把這個位置的 K/V
「寫」進它所在的 block。如果這個 block 是被別人共用的，這一寫就會
弄髒別人的 K/V——這正是 COW 要解決的問題。保留「最後一個 block
永遠私有」這條規則，從根本避免了這個情境，被共用的 block 因此
天生就是唯讀的，完全不需要額外的複製機制。

這是一個真實的工程取捨：真正的 vLLM 也遵循類似的原則（快取比對
的長度一定嚴格小於 prompt 全長，保留至少 1 個 token 需要真的算），
理由完全一樣。
"""

from __future__ import annotations

import xxhash

from mini_vllm.engine.block_manager import BlockManager


class PrefixCachingBlockManager(BlockManager):
    def __init__(self, num_blocks: int, block_size: int) -> None:
        super().__init__(num_blocks, block_size)

        # 改用 set 取代父類別的 deque：需要支援「把某個特定編號的
        # block 從 free pool 裡挖出來」（discard）—— 一個 block 因為
        # 沒人用而回到 free pool 後，只要內容還沒被覆寫，隨時可能被
        # 後來的序列用 hash 命中並「借走」，這時就需要把它從 free
        # pool 裡移除，deque 沒有 O(1) 的按值刪除。
        self.free_block_ids: set[int] = set(range(num_blocks))

        self.ref_counts: dict[int, int] = {}  # 實體 block id -> 目前有幾個序列在用
        self.hash_to_block: dict[bytes, int] = {}  # 內容 hash -> 實體 block id
        self.block_content_hash: dict[int, bytes] = {}  # 實體 block id -> 它現在存的內容 hash（反查用）

    @staticmethod
    def _hash_block(parent_hash: bytes | None, block_tokens: tuple[int, ...]) -> bytes:
        """
        鏈式 hash：這個 block 的 hash，不只跟它自己裝的 token 有關，
        還跟「它前面所有 block 串起來」的 hash 有關（parent_hash）。

        這是必要的：如果只 hash「這個 block 自己的內容」，兩個序列
        即使某個 block 裝的 token 剛好一模一樣，但更前面的 block
        內容不同，也會被誤判成可以共用。鏈式 hash 保證了「共用某個
        block」等於「共用它以及它之前所有 block 代表的完整前綴」——
        這正是 prefix（前綴）caching 這個名字的意義所在。
        """
        h = xxhash.xxh64()
        if parent_hash is not None:
            h.update(parent_hash)
        h.update(repr(block_tokens).encode())
        return h.digest()

    def _take_from_free_pool(self) -> int:
        if not self.free_block_ids:
            raise MemoryError(
                f"KV cache block 池已滿（共 {self.num_blocks} 個 block，"
                f"block_size={self.block_size}），且沒有可以被搶佔的對象。"
            )
        block_id = self.free_block_ids.pop()
        # 這個實體 block 如果之前登記過別的內容 hash，那份索引已經
        # 不再有效了（內容即將被蓋掉）——一定要清掉，不然之後有人
        # 拿舊 hash 來查，會查到這個 block，卻讀到完全不對的內容。
        old_hash = self.block_content_hash.pop(block_id, None)
        if old_hash is not None:
            self.hash_to_block.pop(old_hash, None)
        self.ref_counts[block_id] = 1
        return block_id

    def allocate_prefill(
        self, seq_id: str, prompt_token_ids: list[int]
    ) -> tuple[list[int], int]:
        """
        序列第一次被排進 running 時呼叫，處理完整 prompt 的 block
        配置，並嘗試用內容比對命中 prefix cache。

        回傳: (block_table, num_cached_tokens)
          num_cached_tokens 是「前面有幾個 token 直接借用了別人算好
          的 K/V，不需要重新跑 forward」——呼叫端可以把
          `Sequence.num_computed_tokens` 直接設成這個值，跳過對應
          的 forward 計算，這就是 prefix caching 真正省下時間的地方
          （不只是省記憶體，是省了實際的矩陣運算）。
        """
        table = self.block_tables.setdefault(seq_id, [])
        assert not table, "allocate_prefill 只該在序列第一次配置 block 時呼叫"

        num_blocks_total = self.num_blocks_needed(len(prompt_token_ids))
        # 最後一個邏輯 block 的索引，被排除在「可以比對/註冊快取」的
        # 範圍之外（見檔案開頭「為什麼不需要 COW」的說明）。
        cacheable_upto = num_blocks_total - 1

        parent_hash: bytes | None = None
        num_cached_tokens = 0

        for logical_idx in range(num_blocks_total):
            start = logical_idx * self.block_size
            end = min(start + self.block_size, len(prompt_token_ids))
            block_tokens = tuple(prompt_token_ids[start:end])
            is_full = (end - start) == self.block_size
            eligible = is_full and logical_idx < cacheable_upto

            block_id = None
            content_hash = None
            if eligible:
                content_hash = self._hash_block(parent_hash, block_tokens)
                cached_block_id = self.hash_to_block.get(content_hash)
                if cached_block_id is not None:
                    # cache hit：直接借用這個 block，完全不需要重新
                    # 配置、也不需要重新計算這段 token 的 K/V。
                    block_id = cached_block_id
                    self.ref_counts[block_id] = self.ref_counts.get(block_id, 0) + 1
                    self.free_block_ids.discard(block_id)
                    num_cached_tokens = end

            if block_id is None:
                block_id = self._take_from_free_pool()
                if eligible:
                    # cache miss，但這個 block 屬於「可快取」範圍：
                    # 登記起來，讓下一個開頭一樣的請求可以命中。
                    content_hash = self._hash_block(parent_hash, block_tokens)
                    self.hash_to_block[content_hash] = block_id
                    self.block_content_hash[block_id] = content_hash

            table.append(block_id)
            parent_hash = content_hash if is_full else None

        return table, num_cached_tokens

    def grow_private(self, seq_id: str, num_tokens: int) -> list[int]:
        """
        decode 過程中呼叫，單純按需配置新 block（跟 Stage 2 的
        `BlockManager.ensure_capacity`做一樣的事），不做任何 prefix
        cache 比對或註冊。

        這是刻意的簡化：prefix caching 真正有價值的地方，是「不同
        請求共用的開頭」（例如系統提示詞），而 decode 逐 token 生成
        出來的內容，幾乎不可能被其他請求原封不動命中。把「比對快取」
        這件事限定在 prefill 階段，可以讓程式碼保持簡單，也幾乎
        不損失實際效益。
        """
        table = self.block_tables.setdefault(seq_id, [])
        needed = self.num_blocks_needed(num_tokens)
        while len(table) < needed:
            table.append(self._take_from_free_pool())
        return table

    def free(self, seq_id: str) -> None:
        """
        歸還一個序列持有的所有 block。跟 Stage 2 的 `free()` 不同：
        這裡的「歸還」只是把 ref_count 減 1，只有真的沒有任何序列
        還在用某個 block（ref_count 歸零）時，才會把它放回 free
        pool——而且放回去之後，它的內容 hash 索引依然保留，
        並沒有被清空或覆寫，如果之後有新序列剛好需要一模一樣的
        前綴，這個 block 還是可以被立刻「復活」重新借用。
        """
        table = self.block_tables.pop(seq_id, [])
        for block_id in table:
            self.ref_counts[block_id] = self.ref_counts.get(block_id, 1) - 1
            if self.ref_counts[block_id] <= 0:
                self.ref_counts.pop(block_id, None)
                self.free_block_ids.add(block_id)

    def prefix_cache_stats(self) -> dict:
        """方便觀察/測試/示範目前快取的使用狀況。"""
        return {
            "num_registered_blocks": len(self.hash_to_block),
            "num_free_blocks": self.num_free_blocks,
            "shared_blocks": {
                block_id: ref for block_id, ref in self.ref_counts.items() if ref > 1
            },
        }
