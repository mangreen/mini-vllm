"""
Stage 3：Scheduler —— Continuous Batching 的排程核心。

Stage 0-2 的生成迴圈，都是「一個序列從頭跑到尾才換下一個」。這樣做
的問題：序列的生成長度不一，先做完的序列会讓後面等待的請求白白
浪費時間；反過來說，晚到的請求也得等前面的序列全部做完才能開始。

Continuous batching 的精神很直接：**每一個 engine step，都重新決定
一次「這一輪要跑哪些序列」**——完成的序列立刻讓出位置、新抵達的
請求只要記憶體夠就能馬上加入，不需要等待一整批（batch）的其他序列
都跑完。這支 Scheduler 就是做這個決定的地方。

跟 Stage 2 的關係：Stage 2 的 BlockManager 已經能讓多個序列「先後」
共用同一個 block 池；Scheduler 在這之上加了「排程政策」：
  1. **Admission（准入）**：waiting 裡的新請求，什麼時候可以加入 running？
  2. **Preemption（搶佔）**：記憶體不夠時，該犧牲誰讓給誰？

本階段的搶佔策略採用 vLLM 論文裡提到的兩種做法之一：**recompute**
（丟掉被搶佔序列的 KV cache，之後要重新排進 running 時，從頭
recompute 整段內容）。另一種做法是 **swap**（把 KV cache 換出到
CPU/硬碟，之後再換回來），這裡不實作，因為 recompute 邏輯簡單、
不需要額外的儲存管理，且跟 Stage 2 的 `ensure_capacity` 天然契合。

---

搶佔與歸還過程 ASCII 圖解：

Memory Pool (Block 池容量限制)
┌──────────┬──────────┬──────────┬──────────┐
│ Block 0  │ Block 1  │ Block 2  │ Block 3  │ (全滿狀態)
└──────────┴──────────┴──────────┴──────────┘
  Seq-A 用   Seq-A 用   Seq-B 用   Seq-B 用

當 Seq-A 需要生成下一個 Token，且需要配置第 3 個 Block 時：
1. 發現沒有 free block 了！
2. 觸發 _preempt_one()：選擇排在最後面的 Seq-B 進行搶佔。
3. Seq-B 釋放 Block 2 & Block 3 ➔ 放回 Free Pool。
4. Seq-B 呼叫 reset_for_preemption() ➔ num_computed_tokens 設為 0，塞回 waiting 最前面。
5. Seq-A 拿到空出來的 Block，順利繼續生成！

┌──────────┬──────────┬──────────┬──────────┐
│ Block 0  │ Block 1  │ Block 2  │  (Free)  │
└──────────┴──────────┴──────────┴──────────┘
  Seq-A 用   Seq-A 用   Seq-A 新拿
"""

from __future__ import annotations

from collections import deque

from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    def __init__(self, block_manager: BlockManager) -> None:
        self.block_manager = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        # events：紀錄「最近一次 step() 呼叫」裡實際發生了什麼排程動作
        # （准入、搶佔）。因為 block 是以「一整個 block」為單位配置，
        # 一次搶佔騰出來的空間可能綽綽有餘，導致同一個 step() 裡
        # 「先搶佔、後面又馬上把同一個序列重新排回 running」——單看
        # step() 呼叫前後的 running 集合，會完全看不出中間發生過搶佔。
        # 用一份明確的事件紀錄，才能誠實地觀察到排程器內部實際做了什麼。
        self.events: list[str] = []

    def add_request(self, seq: Sequence) -> None:
        """一個新請求抵達，先排進 waiting 隊伍，下次 step() 才會決定
        它是否能馬上加入 running。"""
        self.waiting.append(seq)

    def _can_grow(self, seq: Sequence) -> bool:
        """
        這個序列如果要處理到『目前已知的全部內容』，block 池夠不夠？
        這個方法只看「目前已知的全部內容」，不會去猜測未來還會不會有新 token 抵達。
        這個方法不會去修改 block 池的狀態，只是純粹「問」。
        這個方法的邏輯很簡單：
          1. 計算這個序列目前已知的全部內容，需要多少個 block。
          2. 看 block 池裡目前有多少個 block 已經被這個序列佔用。
          3. 如果還需要額外的 block，檢查 block 池裡還有沒有空閒的 block。
        """
        needed = self.block_manager.num_blocks_needed(len(seq.all_token_ids))
        have = len(self.block_manager.get_block_table(seq.seq_id))
        extra_needed = max(0, needed - have)
        return extra_needed <= self.block_manager.num_free_blocks

    def _preempt_one(self, candidates: list[Sequence], protect: Sequence) -> Sequence | None:
        """
        從 candidates 裡挑一個犧牲者踢出去，釋放它的 block。

        策略：LIFO（踢掉 candidates 裡排在最後面的序列）。這是所有能
        想到的策略裡最簡單、最容易預測、也最容易寫測試驗證的一種——
        真實的 vLLM 會用更精細的策略（例如優先權、公平性），但排程
        策略本身是可以替換的一個模組，不影響 Scheduler 其他部分的
        設計，這裡先用最簡單的版本把整個機制的骨架搭起來。

        `candidates`：這一輪還「活著」（沒被搶佔過）的序列列表，
        由呼叫端（`step()`）維護——這個方法本身不去碰 `self.running`，
        避免一邊搶佔一邊有外層迴圈也在迭代同一個 list 的問題（見
        `step()` 開頭註解，這是實際踩過的 bug）。
        `protect`：這次呼叫絕對不能踢的序列（正在爭取空間的序列自己）。

        回傳：被搶佔的序列（給呼叫端記錄、從 candidates 中移除），
        如果沒有任何可犧牲的對象則回傳 None。
        """
        for victim in reversed(candidates):
            if victim is protect:
                continue
            self.block_manager.free(victim.seq_id)
            victim.reset_for_preemption()
            # 放回 waiting 隊伍「最前面」而不是最後面：被搶佔的序列
            # 已經做了一部分工作，優先讓它有機會盡快排回 running，
            # 避免它被新抵達的請求無限期插隊、一直排不到（饑餓問題）。
            self.waiting.appendleft(victim)
            self.events.append(f"preempt: {victim.seq_id} 讓給 {protect.seq_id}")
            return victim
        return None

    def step(self) -> list[Sequence]:
        """
        執行一次排程決策，回傳這一輪「確定會被送進模型」的序列列表。

        分兩個階段：
          1. 先照顧 running 裡已經在跑的序列——它們每次都需要多 1 個
             token 的空間，不夠時透過搶佔騰出來。
          2. 再看 waiting 隊伍，把排得到空間的新序列加入 running。

        這個順序是刻意的：**已經在跑的序列優先**，這樣才不會有一個
        序列已經生成到一半，卻因為要讓新請求插隊而被迫中斷。
        """
        self.events = []

        # --- 階段 1：確保 running 裡的序列都能繼續長大 ---
        #
        # 這段邏輯寫的時候踩過一個真實的 bug，值得記錄下來：
        # 一開始用「一邊 for 迴圈迭代 self.running、一邊在搶佔邏輯裡
        # 直接 pop 同一個 self.running」的寫法。問題不只是迭代時
        # 修改 list 本身會亂序，更隱蔽的是：序列 A 在自己的迴圈回合
        # 被判定「可以繼續跑」、加進了 still_running；但後面處理序列 B
        # 時，B 因為記憶體不夠而搶佔了 A——這時候 A 早就已經被加進
        # still_running 了，不會再被移除，於是回傳的 running 列表裡
        # 同時出現「已經被搶佔、block 已被釋放」的 A，一 forward 就
        # 因為 block_table 是空的而炸掉。
        #
        # 修正方式：全程只在區域變數（snapshot、preempted_ids）上操作，
        # 完全不直接修改 self.running，最後才一次性重新賦值，這樣
        # 「搶佔會不會影響到前面已經處理過的序列」這件事，可以用
        # `preempted_ids` 集合誠實地追蹤，不會有時序上的死角。
        #
        # 還有一個更隱蔽的細節：序列 A 可能在「自己的回合」就已經
        # 被判定沒問題、加進了 still_running，但後面處理序列 B 時，
        # B 反過來把 A 搶佔了——這種「先允許、後反悔」的情況，
        # 必須在迴圈跑完之後，用 preempted_ids 對 still_running
        # 做一次最終過濾才抓得到（見迴圈後面那行 list comprehension），
        # 光靠迴圈開頭的 `if seq.seq_id in preempted_ids: continue`
        # 只能擋到「還沒輪到、就已經被搶佔」的序列，擋不到這種「已經
        # 處理完、事後才被搶佔」的序列——這正是第一版留下的殘餘 bug。
        snapshot = list(self.running)
        preempted_ids: set[str] = set()
        still_running: list[Sequence] = []

        for seq in snapshot:
            if seq.seq_id in preempted_ids:
                continue  # 這個序列在前面某一輪，已經被別人的搶佔邏輯犧牲掉了

            candidates = [s for s in snapshot if s.seq_id not in preempted_ids]
            while not self._can_grow(seq):
                victim = self._preempt_one(candidates, protect=seq)
                if victim is None:
                    # 已經沒有別人可以犧牲了，連這個序列自己都保不住
                    # 空間——極端情況：pool 真的太小。把它自己也讓出去。
                    self.block_manager.free(seq.seq_id)
                    seq.reset_for_preemption()
                    self.waiting.appendleft(seq)
                    self.events.append(
                        f"preempt: {seq.seq_id} 連自己都保不住空間（pool 太小）"
                    )
                    preempted_ids.add(seq.seq_id)
                    break
                preempted_ids.add(victim.seq_id)
                candidates = [s for s in candidates if s.seq_id != victim.seq_id]
            else:
                self.block_manager.ensure_capacity(seq.seq_id, len(seq.all_token_ids))
                still_running.append(seq)

        self.running = [s for s in still_running if s.seq_id not in preempted_ids]

        # --- 階段 2：讓排得到空間的新序列加入 running ---
        # 用 FCFS（先到先服務）：如果隊伍最前面的序列都排不進去，
        # 就不去看後面排隊的序列——即使後面有個很短、明明塞得下的序列。
        # 這是為了避免「大請求永遠被插隊的小請求擋住、永遠排不到」的
        # 餓死問題（跟 OS 排程裡的公平性考量是同一類問題）。
        while self.waiting and self._can_grow(self.waiting[0]):
            seq = self.waiting.popleft()
            self.block_manager.ensure_capacity(seq.seq_id, len(seq.all_token_ids))
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            self.events.append(f"admit: {seq.seq_id} 加入 running")

        return self.running

    def finish(self, seq: Sequence) -> None:
        """序列生成完畢，釋放它的 block，離開 running 列表，讓空間給別人。"""
        self.block_manager.free(seq.seq_id)
        if seq in self.running:
            self.running.remove(seq)

    @property
    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting) or bool(self.running)
