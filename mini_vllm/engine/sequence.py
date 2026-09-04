"""
Stage 3：Sequence —— 一個生成請求的狀態機。

Stage 0-2 的範例腳本裡，一個序列從頭生成到尾，中間沒有任何「狀態」
需要被記錄下來（程式的 for 迴圈本身就代表了狀態）。但 continuous
batching 要讓「多個序列交錯執行」——這一個 step 可能只處理序列 A
跟 C，序列 B 在等；下一個 step 序列 B 加進來了，序列 A 卻因為記憶體
不夠被踢出去——這種交錯執行，就需要一個明確的物件把每個序列「目前
進行到哪裡了」記錄下來，而不能再靠 Python 的 for 迴圈隱式地維護。

這就是 Sequence 存在的理由：它是一個請求的「進度存檔」。

狀態轉移：

    WAITING ──排上 running──▶ RUNNING ──生成完 max_new_tokens──▶ FINISHED
       ▲                         │
       └────被搶佔（記憶體不夠）────┘

（被搶佔時 RUNNING -> WAITING，不是 -> FINISHED：這個序列還沒做完，
只是要先讓出記憶體給別人，之後排得到空間會重新排進 running。）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import torch


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Sequence:
    seq_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    output_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING

    # num_computed_tokens：目前 KV cache 裡，已經反映了多少個 token
    # 的內容。跟 Stage 1/2 的 `cache.length` / `KVCache.length` 是同一個
    # 概念，只是搬進 Sequence 裡，因為現在要同時追蹤好幾個序列各自的進度。
    num_computed_tokens: int = 0

    @property
    def all_token_ids(self) -> list[int]:
        """prompt + 目前為止已經生成的內容，合起來就是這個序列『到目前
        為止的完整內容』。"""
        return self.prompt_ids + self.output_ids

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def needs_forward(self) -> bool:
        """還有沒有『沒被算進 cache』的 token 需要處理。"""
        return len(self.all_token_ids) > self.num_computed_tokens

    def next_forward_input(self) -> tuple[torch.Tensor, int]:
        """
        回傳這次該送進模型的 (input_ids, start_pos)。

        這是本階段一個關鍵的統一設計：不管是「序列第一次被排進
        running、要做初次 prefill」「序列已經在跑、正常 decode
        （每次只有 1 個新 token）」，還是「序列被搶佔後重新排進
        running、要 recompute 整個 prompt+已生成內容」，全部都可以
        用同一條公式表示：

            start_pos = num_computed_tokens
            input_ids = all_token_ids[num_computed_tokens:]

        因為 all_token_ids 跟 num_computed_tokens 的差，就是「目前
        還沒反映進 cache」的那一段——
          - 剛加入 running 時，這段是整個 prompt（初次 prefill）。
          - 正常 decode 時，這段只有最新生成的那 1 個 token。
          - 被搶佔重來時，這段是 prompt + 已生成內容的全部（recompute）。
        呼叫端（Scheduler / 生成迴圈）完全不需要為這三種情況各寫一份
        邏輯，統一呼叫這個方法就好。
        """
        ids = self.all_token_ids[self.num_computed_tokens :]
        start_pos = self.num_computed_tokens
        return torch.tensor([ids], dtype=torch.long), start_pos

    def mark_computed(self, num_tokens: int) -> None:
        """一次 forward 呼叫處理完 num_tokens 個 token 後呼叫，
        推進『已經反映進 cache』的進度。"""
        self.num_computed_tokens += num_tokens

    def append_token(self, token_id: int) -> None:
        """把新取樣出來的 token 接到輸出後面；生成滿了就轉成 FINISHED。"""
        self.output_ids.append(token_id)
        if len(self.output_ids) >= self.max_new_tokens:
            self.status = SequenceStatus.FINISHED

    def reset_for_preemption(self) -> None:
        """
        被搶佔時呼叫：這個序列在 KV cache 裡的內容已經被
        BlockManager 釋放掉了，所以『進度存檔』要退回到
        num_computed_tokens=0（下次重新排進 running 時，會用
        recompute 的方式，把 prompt+已生成內容整段重新算一次）。

        注意：已經生成的 output_ids **不會**被丟掉——只是
        「算過的痕跡」被清空，內容本身還在，重新算一次就會拿回
        一模一樣的 K/V（貪婪取樣下數學是確定的），不會改變
        這個序列最終生成出來的文字。
        """
        self.num_computed_tokens = 0
        self.status = SequenceStatus.WAITING
