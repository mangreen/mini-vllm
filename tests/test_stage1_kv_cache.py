import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from mini_vllm.layers.kv_cache import KVCache
from mini_vllm.models.tiny_transformer import TinyTransformer, TinyTransformerConfig
from mini_vllm.models.tiny_transformer_kv import TinyTransformerKV
from mini_vllm.models.tokenizer import CharTokenizer
from baseline_generate import naive_generate
from kv_cache_generate import kv_cache_generate


def make_tokenizer_and_config():
    corpus = "the quick brown fox jumps over the lazy dog "
    tokenizer = CharTokenizer(corpus)
    config = TinyTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=64,
    )
    return tokenizer, config


def make_matched_models(config: TinyTransformerConfig):
    """
    建立一組「架構相同、權重也相同」的 Stage 0 / Stage 1 模型。

    這是驗證 KV cache 實作正確與否的關鍵手法：如果 Stage 1 的
    數學跟 Stage 0 完全等價，那把同一組權重分別載入兩個模型，
    輸出就必須逐數值相等（不是「差不多」，是 allclose 等級的相等）。
    """
    model_v0 = TinyTransformer(config)
    model_v1 = TinyTransformerKV(config)
    model_v1.load_state_dict(model_v0.state_dict())
    model_v0.eval()
    model_v1.eval()
    return model_v0, model_v1


# ---------------------------------------------------------------------------
# KVCache 本身的單元測試
# ---------------------------------------------------------------------------

def test_kv_cache_write_and_read_shapes():
    _, config = make_tokenizer_and_config()
    cache = KVCache(config, batch_size=1)

    k = torch.randn(1, config.n_heads, 5, config.head_dim)
    v = torch.randn(1, config.n_heads, 5, config.head_dim)
    cache.write(layer_idx=0, start_pos=0, k=k, v=v)
    cache.advance(5)

    k_read, v_read = cache.read(layer_idx=0, end_pos=5)
    assert k_read.shape == (1, config.n_heads, 5, config.head_dim)
    assert torch.equal(k_read, k)
    assert torch.equal(v_read, v)


def test_kv_cache_incremental_write():
    """模擬 prefill(3 個 token) + decode(逐個寫入 2 次) 的寫入流程。"""
    _, config = make_tokenizer_and_config()
    cache = KVCache(config, batch_size=1)

    k_prefill = torch.randn(1, config.n_heads, 3, config.head_dim)
    v_prefill = torch.randn(1, config.n_heads, 3, config.head_dim)
    cache.write(layer_idx=0, start_pos=0, k=k_prefill, v=v_prefill)
    cache.advance(3)

    k_step1 = torch.randn(1, config.n_heads, 1, config.head_dim)
    v_step1 = torch.randn(1, config.n_heads, 1, config.head_dim)
    cache.write(layer_idx=0, start_pos=cache.length, k=k_step1, v=v_step1)
    cache.advance(1)

    k_all, v_all = cache.read(layer_idx=0, end_pos=cache.length)
    assert k_all.shape == (1, config.n_heads, 4, config.head_dim)
    assert torch.equal(k_all[:, :, :3], k_prefill)
    assert torch.equal(k_all[:, :, 3:4], k_step1)


def test_kv_cache_overflow_raises():
    _, config = make_tokenizer_and_config()
    cache = KVCache(config, batch_size=1)

    too_long = config.max_seq_len + 1
    k = torch.randn(1, config.n_heads, too_long, config.head_dim)
    v = torch.randn(1, config.n_heads, too_long, config.head_dim)

    try:
        cache.write(layer_idx=0, start_pos=0, k=k, v=v)
        assert False, "應該要因為超過 max_seq_len 而丟出 ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 正確性驗證：Stage 1 必須跟 Stage 0 數值上完全等價
# ---------------------------------------------------------------------------

def test_full_prefill_matches_stage0_forward_exactly():
    """
    只做「一次性 prefill」（不做任何 decode），
    此時 Stage 1 的計算路徑應該跟 Stage 0 的一次性 forward
    數學上完全相同（causal mask 公式在 start_pos=0 時會退化成
    跟 Stage 0 一樣的下三角矩陣），因此 logits 必須逐數值相等。
    """
    tokenizer, config = make_tokenizer_and_config()
    model_v0, model_v1 = make_matched_models(config)

    ids = torch.tensor([tokenizer.encode("the quick brown")])

    with torch.no_grad():
        logits_v0 = model_v0(ids)

        cache = KVCache(config, batch_size=1)
        logits_v1 = model_v1(ids, kv_cache=cache, start_pos=0)

    assert torch.allclose(logits_v0, logits_v1, atol=1e-5)


def test_prefill_plus_decode_matches_stage0_generation_exactly():
    """
    最重要的一個測試：完整跑一次「prefill + 逐 token decode」，
    在貪婪取樣下，生成出來的文字必須跟 Stage 0 的樸素生成迴圈
    逐字元相同。這是驗證 KV cache 整個實作沒有引入任何錯誤的
    最終標準——如果這個測試過了，代表 Stage 1 的優化是「純粹的
    加速」，沒有改變模型的行為。
    """
    tokenizer, config = make_tokenizer_and_config()
    model_v0, model_v1 = make_matched_models(config)

    prompt = "the quick "
    max_new_tokens = 15

    text_v0, _ = naive_generate(model_v0, tokenizer, prompt, max_new_tokens)
    text_v1, _ = kv_cache_generate(model_v1, tokenizer, prompt, max_new_tokens)

    assert text_v0 == text_v1


def test_kv_cache_generate_produces_correct_length():
    tokenizer, config = make_tokenizer_and_config()
    model = TinyTransformerKV(config)

    prompt = "the quick "
    max_new_tokens = 9
    text, step_times = kv_cache_generate(model, tokenizer, prompt, max_new_tokens)

    assert len(text) == len(prompt) + max_new_tokens
    # +1 是因為 step_times 包含 prefill 那一步，decode 只佔 max_new_tokens - 1 步
    assert len(step_times) == max_new_tokens
