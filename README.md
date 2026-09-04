# mini-vllm

從零開始、step by step 實作一個超精簡 vLLM，參考專案：[nano-vllm-lite](https://github.com/pzsacc/nano-vllm-lite)。

執行環境：Intel Mac（無 NVIDIA GPU）、CPU-only PyTorch。直接用載入 Qwen3-0.6B 這種真的「小模型」（選項 B）——它對你的 CPU 來說其實不小。因此模型採用**自己寫的迷你 Transformer**（選項 A），不依賴任何 GPU 專屬功能，確保每個階段都能在本機完整驗證機制正確性。

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
│   ├── layers/         # 實現極致優化、鍵值快取（KV Cache）、支援分頁（Paging）的自注意力機制運算核心
│   ├── engine/         # 實現整個推理生命週期的控管、請求調度與記憶體管理
│   └── ...             
├── examples/           # 可直接執行的示範腳本
├── tests/              # pytest 單元測試
└── requirements.txt
```

## 目前進度

- [x] Stage 0：樸素 baseline（無 KV cache 的 for-loop 生成）— branch `stage-0-baseline`
- [x] Stage 1：手刻 KV Cache — branch `stage-1-kv-cache`
- [x] Stage 2：PagedAttention（Block-based KV Cache）— branch `stage-2-paged-attention`
- [x] Stage 3：Continuous Batching + Scheduler — branch `stage-3-continuous-batching`
- [x] Stage 4：Prefix Caching — branch `stage-4-prefix-caching`
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

詳細說明見 [`docs/stage0-baseline.md`](docs/stage0-baseline.md)、[`docs/stage1-kv-cache.md`](docs/stage1-kv-cache.md)、[`docs/stage2-paged-attention.md`](docs/stage2-paged-attention.md)、[`docs/stage3-continuous-batching.md`](docs/stage3-continuous-batching.md)、[`docs/stage4-prefix-caching.md`](docs/stage4-prefix-caching.md)。

## 執行 Stage 1

```bash
python examples/kv_cache_generate.py
pytest tests/ -v
```

## 執行 Stage 2

```bash
python examples/paged_attention_generate.py
pytest tests/ -v
```

## 執行 Stage 3

```bash
python examples/continuous_batching_generate.py
pytest tests/ -v
```

## 執行 Stage 4

```bash
python examples/prefix_caching_generate.py
pytest tests/ -v
```
