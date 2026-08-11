"""
極簡字元級 tokenizer。

Stage 0 的重點是「推論引擎的機制」，不是「tokenizer 本身」，
所以刻意用最簡單的字元級實作：每個字元就是一個 token。
好處：vocab 很小（幾十到一百多），CPU 上跑起來完全沒有負擔，
而且可以直接肉眼看輸出字串，方便驗證生成邏輯對不對。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CharTokenizer:
    """
    從一段語料建立字元級詞表。這是一個「文字編碼與解碼字典」。
    電腦只懂數字，不懂字元；這個類別負責把輸入的字串拆成單個字母，給每個字母一個編號，並提供雙向翻譯功能。
    
    [文字語料庫 "cat"] ---> 提取不重複字元按順序排 ['a', 'c', 't']
                         │
                         ├─> stoi (字轉數字): {'a': 0, 'c': 1, 't': 2}
                         └─> itos (數轉文字): {0: 'a', 1: 'c', 2: 't'}

    ex.
    ```python
    tokenizer = CharTokenizer(corpus="aba")
    # 自動生成：
    # tokenizer.stoi -> {'a': 0, 'b': 1}
    # tokenizer.itos -> {0: 'a', 1: 'b'}
    ```
    """

    corpus: str # 文字語料庫
    stoi: dict[str, int] = field(default_factory=dict, init=False) # 字轉數字字典
    itos: dict[int, str] = field(default_factory=dict, init=False) # 數轉文字字典

    def __post_init__(self) -> None:
        """
        從語料庫中提取不重複字元，並建立 stoi 與 itos 字典。
        """
        chars = sorted(set(self.corpus))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        """
        詞表大小 = 不重複字元數量

        ex. 若 stoi 為 {'a': 0, 'b': 1}，則 vocab_size 回傳 2。
        """
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        """
        將字串轉成對應的數字列表。
        若輸入字串中有不在詞表中的字元，會丟出 ValueError。

        ex.
        ```python
        # 文字 "ab"  ──[ 查 stoi 字典 ]──>  [0, 1]

        tokenizer = CharTokenizer(corpus="aba")
        tokenizer.encode("ab")  # 回傳 [0, 1]
        tokenizer.encode("ac")  # ValueError: 字元 {'c'} 不在詞表中，請確認語料涵蓋了你要編碼的所有字元。
        ```
        """
        unknown = set(text) - set(self.stoi)
        if unknown:
            raise ValueError(
                f"字元 {unknown} 不在詞表中，請確認語料涵蓋了你要編碼的所有字元。"
            )
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """
        將電腦吐出來的數字清單，重新拼回人類看得懂的文字。
        若輸入數字列表中有不在詞表中的數字，會丟出 ValueError。

        ex.
        ```python
        # 數字 [0, 1]  ──[ 查 itos 字典 ]──>  "ab"
        tokenizer = CharTokenizer(corpus="aba")
        tokenizer.decode([0, 1])  # 回傳 "ab"
        tokenizer.decode([0, 2])  # ValueError: 數字 {2} 不在詞表中，請確認輸入的數字列表涵蓋了你要解碼的所有數字。     
        ```
        """
        unknown = set(ids) - set(self.itos)
        if unknown:
            raise ValueError(
                f"數字 {unknown} 不在詞表中，請確認輸入的數字列表涵蓋了你要解碼的所有數字。"
            )
        return "".join(self.itos[i] for i in ids)
