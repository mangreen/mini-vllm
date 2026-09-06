"""
Stage 5：ChunkedPrefillScheduler —— 把長 prefill 切成小塊，
跟 decode 混合排程（decode-first mixed scheduling，也稱
Sarathi-style scheduling）。

## 這階段要解決的問題

Stage 3 的 Scheduler 有一個沒被講出來的假設：**一個序列只要被排進
running，這一步就會把它「所有還沒算過的內容」一次處理完**（見
`Sequence.next_forward_input()` 沒有長度上限）。這在序列都很短的
時候沒問題，但如果有一個很長的 prefill（例如幾千個 token 的
prompt）混在其他正在 decode 的序列之間，會發生：

    這個 engine step 除了要幫其他序列各生出 1 個新 token，
    還要花很多時間把這個超長 prefill 整段算完
    -> 這一步的實際耗時被這個長 prefill 拖得很長
    -> 其他序列的使用者，這一步等到新 token 的時間變得很長

這正是「一個很長的 prefill 會拖累其他序列 decode 的即時性」這個
問題——在 TPOT（time-per-output-token，使用者感受到的逐字輸出
速度）上看起來就是一個時好時壞的延遲尖峰。

## 解法：Chunked Prefill

把一次很長的 prefill，切成好幾個固定大小的「chunk」，分散到好幾個
engine step 慢慢做完，而且**每一步都優先把 budget 留給 decode**，
剩下的 budget 才分給正在 chunked prefill 的序列。這樣一來，不管
有沒有一個超長的 prefill 正在背景進行，其他序列每一步都還是能穩定
拿到它的 decode token。
"""

from __future__ import annotations

from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.engine.scheduler import Scheduler
from mini_vllm.engine.sequence import Sequence


class ChunkedPrefillScheduler(Scheduler):
    def __init__(
        self,
        block_manager: BlockManager,
        token_budget_per_step: int,
        max_prefill_chunk: int,
    ) -> None:
        """
        token_budget_per_step：這一個 engine step，總共最多處理幾個
            token（不管是 decode 還是 prefill chunk，都算在這個
            budget 裡）——對應真實 vLLM 裡的
            `max_num_batched_tokens`。
        max_prefill_chunk：單一序列，單一個 step，最多可以處理幾個
            「prefill 型」的 token——這是「chunked」真正的來源：
            就算 budget 很充裕，一個序列也不能一次把一整段很長的
            prefill 都做完，必須分成好幾步。
        """
        super().__init__(block_manager)
        self.token_budget_per_step = token_budget_per_step
        self.max_prefill_chunk = max_prefill_chunk

    def step(self) -> list[tuple[Sequence, int]]:
        """
        跟 Stage 3 的 `Scheduler.step()` 分工不同：那邊決定的是
        「block 記憶體層次」的准入/搶佔（哪些序列這一步能不能跑，
        完全跟要處理幾個 token 無關）；這裡在那之上，多決定一件事：
        「這一步，每個能跑的序列，各自要處理多少個 token」。

        回傳：[(seq, num_tokens_this_step), ...]，只包含這一步真的
        會被送進模型的序列，`num_tokens_this_step` 保證 >= 1。
        """
        running = super().step()  # 沿用 Stage 3 的准入/搶佔邏輯，不變

        decode_ready: list[Sequence] = []
        prefill_pending: list[Sequence] = []
        for seq in running:
            pending = seq.pending_token_count
            if pending <= 0:
                continue  # 理論上不會發生（見 Sequence 的不變量），保留防呆
            elif pending == 1:
                decode_ready.append(seq)
            else:
                prefill_pending.append(seq)

        schedule: list[tuple[Sequence, int]] = []
        budget = self.token_budget_per_step

        # 階段 1：decode 優先——它只需要 1 個 token 就能讓使用者拿到
        # 下一個字，budget 幾乎用不了多少，卻是最直接影響使用者體感
        # 延遲的地方，所以永遠先滿足它。
        for seq in decode_ready:
            if budget <= 0:
                break  # 極端情況：budget 小到連 1 個 decode token 都塞不下
            schedule.append((seq, 1))
            budget -= 1

        # 階段 2：剩下的 budget 才分給還在（chunked）prefill 的序列，
        # 每個序列這一步最多只吃 `max_prefill_chunk` 個 token——
        # 這正是「chunked」的來源：一個很長的 prefill，會被切成
        # 好幾個 step 慢慢做完，而不是一次把整批 decode 序列卡住。
        for seq in prefill_pending:
            if budget <= 0:
                break
            allocated = min(budget, seq.pending_token_count, self.max_prefill_chunk)
            if allocated > 0:
                schedule.append((seq, allocated))
                budget -= allocated

        return schedule
