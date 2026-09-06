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

        # 實體 block id -> 目前有幾個序列在用
        # 這個計數只用來判斷「這個 block 是否可以放回 free pool」，
        # 不用來判斷「這個 block 是否可以被借用」——
        # 只要它的內容 hash 還在 hash_to_block 裡，就可以被借用，
        # 不管 ref_count 是多少。
        # ex. Bock5 被三個序列共用： ref_counts[5] = 3
        # ReqA ─┐
        # ReqB ─┼─> Block5
        # ReqC ─┘  
        self.ref_counts: dict[int, int] = {}

        # 內容 hash -> 實體 block id
        # 這個索引只用來判斷「這個 block 是否可以被借用」——
        # 只要它的內容 hash 還在這裡，就可以被借用，不管它的實體 block id 是多少。
        self.hash_to_block: dict[bytes, int] = {}  

        # 實體 block id -> 它現在存的內容 hash（反查用）
        # 這個索引只用來「當某個 block 被搶佔時，清掉它舊的 hash 索引」，
        # 不用來判斷「這個 block 是否可以被借用」——
        # 只要它的內容 hash 還在 hash_to_block 裡，就可以被借用，不管它的實體 block id 是多少。
        self.block_content_hash: dict[int, bytes] = {}  

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

        ex.
            SeqA: [A B C D] [E F G H] [I J K L]
            SeqB: [A B C D] [E F G H] [X Y Z]
            SeqC: [M N O P] [E F G H] [Q R S T]

        SeqA 和 SeqB 的前兩個 block 都一模一樣，SeqC 的第一個 block就不同了。鏈式 hash 保證了：
            - SeqA 的第二個 block 的 hash = hash(hash([A B C D]), [E F G H])
            - SeqB 的第二個 block 的 hash = hash(hash([A B C D]), [E F G H])
            - SeqC 的第二個 block 的 hash = hash(hash([M N O P]), [E F G H])
        這三個 hash 彼此不同，SeqC 的第二個 block 不會被誤判成可以跟 SeqA/SeqB 共用。
        """
        h = xxhash.xxh64()
        if parent_hash is not None: # 如果這個 block 前面有其他 block，納入前面所有 block 的 hash
            h.update(parent_hash)
        h.update(repr(block_tokens).encode()) # 把這個 block 的內容納入計算
        return h.digest() # 回傳 8 bytes 的 hash

    def _take_from_free_pool(self) -> int:
        """
        從 free pool 拿一個 block，並把它的 ref_count 設成 1。
        這個 block 可能是「全新」的，也可能是「之前被別的序列用過、但現在沒人用」的 block——
        不管怎樣，拿到之後都要把它的舊 hash 索引清掉，不然之後有人拿舊 hash 來查，
        會查到這個 block，卻讀到完全不對的內容。
        """ 
        if not self.free_block_ids:
            raise MemoryError(
                f"KV cache block 池已滿（共 {self.num_blocks} 個 block，"
                f"block_size={self.block_size}），且沒有可以被搶佔的對象。"
            )
        block_id = self.free_block_ids.pop()
        # 這個實體 block 如果之前登記過別的內容 hash，那份索引已經
        # 不再有效了（內容即將被蓋掉）——一定要清掉，不然之後有人
        # 拿舊 hash 來查，會查到這個 block，卻讀到完全不對的內容。
        # ex. 
        # Block5 之前被 SeqA 用過，登記了 hashA，
        # 但現在 SeqB 搶走它，把它的內容改成 hashB，
        # 這時就要把 hashA -> Block5 的索引清掉，
        # 否則 SeqA 之後再來查會查到 Block5，卻讀到完全不對的內容。
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

        # 這個序列的 block table（實體 block 編號列表）
        # 這個 table 會被存進 self.block_tables[seq_id]，之後 decode 過程中
        # 會直接用這個 table 來對照實體 block。
        # 作用： 檢查字典 self.block_tables 中是否已經存在鍵值 seq_id。
        # - 如果已經存在（之前已經為這個序列配置過區塊）： 
        #   - 它不會修改字典，而是直接返回該 seq_id 已經對應的列表（裡面通常已經有舊的 Block 資料）。
        # - 如果不存在（第一次遇到這個序列）： 
        #   - 它會在字典中新增 seq_id，並將其值設為一個空的列表 []，然後返回這個空列表。
        table = self.block_tables.setdefault(seq_id, [])

        # 這個方法只該在序列第一次配置 block 時呼叫，之後 decode 過程中
        # 只會呼叫 grow_private()，不會再呼叫 allocate_prefill
        # 這個 assert 是為了避免程式碼誤用，造成「同一個序列的前綴被比對快取」的情境。
        # 這個情境不會出現，因為「前綴」是指 prompt 的內容，而 decode 過程中產生的 token 幾乎不可能被其他序列原封不動命中，這也是為什麼把「比對快取」限定在 prefill 階段的原因。
        # - 如果 table 是空的（True）： 
        #   - 代表這個序列確實是第一次配置，斷言成功，程式繼續往下執行。
        # - 如果 table 不是空的（False）： 
        #   - 代表這個序列已經配置過 block，斷言失敗，程式會拋出 AssertionError，
        #   - 並顯示訊息 "allocate_prefill 只該在序列第一次配置 block 時呼叫"。    
        assert not table, "allocate_prefill 只該在序列第一次配置 block 時呼叫"

        # 這個 prompt 需要幾個 block 才能存下所有 token
        # ex. 
        # prompt_token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # block_size = 4
        # num_blocks_total = ceil(9 / 4) = 3
        # - 這個 prompt 的邏輯 block 分布：
        #   - block 0: [1, 2, 3, 4]
        #   - block 1: [5, 6, 7, 8]
        #   - block 2: [9] (最後一個 block 只用了 1 格，其他 3 格空著)
        num_blocks_total = self.num_blocks_needed(len(prompt_token_ids))
        
        # 最後一個邏輯 block 的索引，被排除在「可以比對/註冊快取」的
        # 範圍之外（見檔案開頭「為什麼不需要 COW」的說明）。
        # ex.
        # - num_blocks_total = 3
        # - cacheable_upto = 2
        #   - 這個 prompt 的邏輯 block 分布：
        #     - block 0: [1, 2, 3, 4] (可以比對/註冊快取)
        #     - block 1: [5, 6, 7, 8] (可以比對/註冊快取)
        #     - block 2: [9] (最後一個 block，不可以比對/註冊快取)
        cacheable_upto = num_blocks_total - 1

        parent_hash: bytes | None = None
        num_cached_tokens = 0

        # 這個迴圈的邏輯：
        # 1. 把 prompt 分成一個個 block（每個 block 裝 block_size 個 token，最後一個 block 可能不滿）。
        # 2. 對每個 block，先檢查它是否「可快取」（不是最後一個 block，且裝滿 block_size 個 token）。
        # 3. 如果可快取，計算它的 hash，檢查 hash_to_block 是否有對應的 block_id：
        #    - 如果有，表示命中快取，直接借用這個 block（不用重新配置，也不用重新計算 K/V），並增加 ref_count。
        #    - 如果沒有，表示快取未命中，從 free pool 拿一個新的 block，並把它的 hash 登記進 hash_to_block。
        # 4. 如果不可快取（最後一個 block 或不滿 block_size），直接從 free pool 拿一個新的 block，不做快取登記。
        # 5. 把每個 block 的實體 block_id 加入 table，最後回傳 table 和 num_cached_tokens。
        # ex.
        # - prompt_token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # - block_size = 4
        # - num_blocks_total = 3
        # - cacheable_upto = 2
        # - 迴圈過程：
        #   - logical_idx = 0:
        #     - start = 0, end = 4, block_tokens = (1, 2, 3, 4), is_full = True, eligible = True
        #     - 計算 hash，檢查 hash_to_block，假設沒有命中，從 free pool 拿 Block5，登記 hash -> Block5，table = [5]
        #   - logical_idx = 1:
        #     - start = 4, end = 8, block_tokens = (5, 6, 7, 8), is_full = True, eligible = True
        #     - 計算 hash，檢查 hash_to_block，假設命中 Block2，table = [5, 2]，num_cached_tokens = 8
        #  - logical_idx = 2:
        #     - start = 8, end = 9, block_tokens = (9,), is_full = False, eligible = False
        #     - 從 free pool 拿 Block7，table = [5, 2, 7]
        #   - 回傳 table = [5, 2, 7], num_cached_tokens = 8
        for logical_idx in range(num_blocks_total):
            start = logical_idx * self.block_size
            end = min(start + self.block_size, len(prompt_token_ids))
            block_tokens = tuple(prompt_token_ids[start:end])
            is_full = (end - start) == self.block_size
            eligible = is_full and logical_idx < cacheable_upto # 滿足「可快取」的條件：1. 不是最後一個 block，且 2. Full block（裝滿 block_size 個 token）

            block_id = None
            content_hash = None

            # 如果這個 block 屬於「可快取」範圍，先檢查 hash_to_block 是否有對應的 block_id：
            if eligible:
                content_hash = self._hash_block(parent_hash, block_tokens)
                cached_block_id = self.hash_to_block.get(content_hash)

                # cahce hit: 如果命中快取，直接借用這個 block（不用重新配置，也不用重新計算 K/V），並增加 ref_count。
                if cached_block_id is not None:
                    # cache hit：直接借用這個 block，完全不需要重新
                    # 配置、也不需要重新計算這段 token 的 K/V。
                    block_id = cached_block_id

                    # 增加這個 block 的 ref_count，表示又多了一個序列在用它
                    self.ref_counts[block_id] = self.ref_counts.get(block_id, 0) + 1 

                    # 從 free pool 移除這個 block，避免被其他序列搶走
                    self.free_block_ids.discard(block_id) 

                    # 記錄這個序列有多少 token 是直接借用別人算好的 K/V，不需要重新跑 forward
                    num_cached_tokens = end

            # chache miss：從 free pool 拿一個新的 block，並把它的 hash 登記進 hash_to_block（如果 eligible）。
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
        """debug/demo/test 工具，方便觀察/測試/示範目前快取的使用狀況。"""
        return {
            "num_registered_blocks": len(self.hash_to_block),
            "num_free_blocks": self.num_free_blocks,
            "shared_blocks": {
                block_id: ref for block_id, ref in self.ref_counts.items() if ref > 1
            },
        }
