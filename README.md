# mini-vllm

從零開始、step by step 實作一個超精簡 vLLM，參考專案：[nano-vllm-lite](https://github.com/pzsacc/nano-vllm-lite)。

執行環境：Intel Mac（無 NVIDIA GPU）、CPU-only PyTorch。直接用載入 Qwen3-0.6B 這種真的「小模型」——它對你的 CPU 來說其實不小（選項 B）。因此模型採用**自己寫的迷你 Transformer**（選項 A），不依賴任何 GPU 專屬功能，確保每個階段都能在本機完整驗證機制正確性。

## Git 工作流程

- 每個學習階段一個獨立 branch：`stage-0-baseline`、`stage-1-kv-cache`、`stage-2-paged-attention` …
- 每個 branch 完成後合併回 `main`，並在 `docs/` 底下留一份該階段的學習筆記 + 架構流程圖。
- `.gitignore` 已排除虛擬環境、模型權重、log、cache 等不需要進版控的檔案。

## 目錄結構

```bash
mini-vllm/
├── docs/               # 每階段的學習重點、說明、架構流程圖（mermaid）
├── mini_vllm/
│   ├── models/         # 自製迷你 Transformer
│   └── ...             # 後續階段會逐步加入 engine/、layers/
├── examples/            # 可直接執行的示範腳本
├── tests/              # pytest 單元測試
└── requirements.txt
```

## 目前進度

- [x] Stage 0：樸素 baseline（無 KV cache 的 for-loop 生成）— branch `stage-0-baseline`
- [ ] Stage 1：手刻 KV Cache
- [ ] Stage 2：PagedAttention（Block-based KV Cache）
- [ ] Stage 3：Continuous Batching + Scheduler
- [ ] Stage 4：Prefix Caching
- [ ] Stage 5：Chunked Prefill
- [ ] Stage 6：GPU 加速層（概念閱讀 / 選用雲端 GPU）
- [ ] Stage 7：完整引擎 + API

## 安裝

```bash
pip install -r requirements.txt
```

## 執行 Stage 0

```bash
python examples/baseline_generate.py
pytest tests/
```

詳細說明見 [`docs/stage0-baseline.md`](docs/stage0-baseline.md)。
