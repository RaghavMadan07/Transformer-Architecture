import pytest
import torch

from modern_transformer import ByteTokenizer, Transformer, TransformerConfig
from modern_transformer.layers import apply_rotary_emb, build_rope_cache
from modern_transformer.layers.attention import build_attention_mask, repeat_kv


def make(cfg, seed=0):
    torch.manual_seed(seed)
    return Transformer(cfg).eval()


def test_forward_shapes(cfg):
    m = make(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    out = m(x)
    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert out.loss is None


def test_loss_and_backward(cfg):
    m = make(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    out = m(x, targets=x)
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    out.loss.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_causality(cfg):
    m = make(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 12))
    y = x.clone()
    y[0, 8:] = (y[0, 8:] + 1) % cfg.vocab_size
    lx, ly = m(x).logits, m(y).logits
    assert torch.allclose(lx[:, :8], ly[:, :8], atol=1e-5)
    assert not torch.allclose(lx[:, 8:], ly[:, 8:], atol=1e-5)


@pytest.mark.parametrize("window", [None, 4])
def test_kv_cache_matches_full_forward(cfg, window):
    cfg.sliding_window = window
    m = make(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 20))
    full = m(x).logits

    out = m(x[:, :10], use_cache=True)
    past, chunks = out.past_key_values, [out.logits]
    for i in range(10, 20):
        out = m(x[:, i : i + 1], past_key_values=past, use_cache=True)
        past = out.past_key_values
        chunks.append(out.logits)
    assert torch.allclose(full, torch.cat(chunks, dim=1), atol=1e-4)


def test_sliding_window_limits_receptive_field():
    mask = build_attention_mask(6, 6, torch.device("cpu"), window=3)[0, 0]
    assert mask[5].tolist() == [False, False, False, True, True, True]
    assert mask[0].tolist() == [True] + [False] * 5


def test_padding_mask_no_nans(cfg):
    m = make(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.long)
    mask[0, :4] = 0  # left padding
    assert torch.isfinite(m(x, attention_mask=mask).logits).all()


def test_gqa_repeat_kv():
    x = torch.randn(2, 2, 5, 8)
    y = repeat_kv(x, 3)
    assert y.shape == (2, 6, 5, 8)
    assert torch.equal(y[:, 0], y[:, 2]) and torch.equal(y[:, 3], y[:, 5])


def test_rope_preserves_norm_and_is_relative():
    cos, sin = build_rope_cache(16, 32)
    q = torch.randn(1, 1, 8, 16)
    k = torch.randn(1, 1, 8, 16)
    q2, k2 = apply_rotary_emb(q, k, cos[:8], sin[:8])
    assert torch.allclose(q.norm(dim=-1), q2.norm(dim=-1), atol=1e-5)
    # dot products depend only on relative offset: shift all positions by 5
    q3, k3 = apply_rotary_emb(q, k, cos[5:13], sin[5:13])
    s1 = q2 @ k2.transpose(-1, -2)
    s2 = q3 @ k3.transpose(-1, -2)
    assert torch.allclose(s1, s2, atol=1e-4)


def test_moe_layers_and_aux_loss(moe_cfg):
    m = make(moe_cfg).train()
    assert [b.is_moe for b in m.blocks] == [False, True, False, True]
    x = torch.randint(0, moe_cfg.vocab_size, (4, 16))
    out = m(x, targets=x)
    assert out.aux_loss > 0 and torch.isfinite(out.loss)
    out.loss.backward()
    assert m.blocks[1].ffn.gate.weight.grad is not None


def test_weight_tying(cfg):
    m = make(cfg)
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()
    cfg2 = TransformerConfig.from_dict({**cfg.to_dict(), "tie_embeddings": False})
    m2 = make(cfg2)
    assert m2.lm_head.weight.data_ptr() != m2.tok_emb.weight.data_ptr()


def test_gradient_checkpointing_matches(cfg):
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    m1 = make(cfg).train()
    cfg2 = TransformerConfig.from_dict({**cfg.to_dict(), "gradient_checkpointing": True})
    m2 = make(cfg2).train()
    m2.load_state_dict(m1.state_dict())
    l1 = m1(x, targets=x).loss
    l2 = m2(x, targets=x).loss
    l1.backward(); l2.backward()
    assert torch.allclose(l1, l2, atol=1e-5)
    assert torch.allclose(m1.tok_emb.weight.grad, m2.tok_emb.weight.grad, atol=1e-4)


def test_generate(cfg):
    m = make(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (2, 5))
    out = m.generate(prompt, max_new_tokens=10, temperature=0.9, top_k=20, top_p=0.9)
    assert out.shape == (2, 15)
    greedy1 = m.generate(prompt, max_new_tokens=8, temperature=0)
    greedy2 = m.generate(prompt, max_new_tokens=8, temperature=0)
    assert torch.equal(greedy1, greedy2)


def test_generate_respects_context_limit(cfg):
    m = make(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len - 3))
    assert m.generate(prompt, max_new_tokens=50).shape[1] == cfg.max_seq_len


def test_too_long_input_raises(cfg):
    m = make(cfg)
    with pytest.raises(ValueError):
        m(torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len + 1)))


def test_config_validation_and_roundtrip(tmp_path):
    with pytest.raises(ValueError):
        TransformerConfig(d_model=100, n_heads=8)
    with pytest.raises(ValueError):
        TransformerConfig(n_heads=8, n_kv_heads=3)
    c = TransformerConfig(d_model=128, n_heads=4, n_kv_heads=2)
    c.save(tmp_path / "c.json")
    assert TransformerConfig.load(tmp_path / "c.json") == c


def test_tokenizer_roundtrip():
    t = ByteTokenizer()
    s = "Héllo, 世界!"
    assert t.decode(t.encode(s, add_bos=True, add_eos=True)) == s
