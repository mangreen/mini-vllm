import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from mini_vllm.models.tiny_transformer import TinyTransformer, TinyTransformerConfig
from mini_vllm.models.tokenizer import CharTokenizer
from baseline_generate import naive_generate


def make_model_and_tokenizer():
    corpus = "the quick brown fox jumps over the lazy dog "
    tokenizer = CharTokenizer(corpus)
    config = TinyTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=64,
    )
    model = TinyTransformer(config)
    return model, tokenizer, config


def test_forward_output_shape():
    model, tokenizer, config = make_model_and_tokenizer()
    ids = torch.tensor([tokenizer.encode("the quick")])
    logits = model(ids)
    assert logits.shape == (1, len(tokenizer.encode("the quick")), tokenizer.vocab_size)


def test_causal_mask_blocks_future_tokens():
    """
    驗證 causal mask 真的生效：改變序列「後面」的 token，
    不應該影響「前面」位置的 logits（否則就是偷看了未來）。
    """
    model, tokenizer, config = make_model_and_tokenizer()
    model.eval()

    base = "the quick brown"
    ids_a = torch.tensor([tokenizer.encode(base)])

    modified = base[:-1] + "k"  # 改最後一個字元
    ids_b = torch.tensor([tokenizer.encode(modified)])

    with torch.no_grad():
        logits_a = model(ids_a)
        logits_b = model(ids_b)

    # 除了最後一個位置，前面所有位置的 logits 應該完全相同
    assert torch.allclose(logits_a[0, :-1, :], logits_b[0, :-1, :], atol=1e-6)


def test_naive_generate_is_deterministic_with_greedy_sampling():
    model, tokenizer, config = make_model_and_tokenizer()

    torch.manual_seed(42)
    text_1, _ = naive_generate(model, tokenizer, "the quick ", max_new_tokens=5)

    torch.manual_seed(42)
    text_2, _ = naive_generate(model, tokenizer, "the quick ", max_new_tokens=5)

    assert text_1 == text_2


def test_naive_generate_produces_correct_length():
    model, tokenizer, config = make_model_and_tokenizer()
    prompt = "the quick "
    max_new_tokens = 7

    text, step_times = naive_generate(model, tokenizer, prompt, max_new_tokens)

    assert len(text) == len(prompt) + max_new_tokens
    assert len(step_times) == max_new_tokens
