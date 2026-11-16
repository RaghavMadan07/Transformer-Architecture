import math

import pytest
import torch
import torch.nn.functional as F

from seq2seq_transformer import Transformer, TransformerConfig, greedy_decode
from seq2seq_transformer.attention import MultiHeadAttention, scaled_dot_product_attention
from seq2seq_transformer.data import Seq2SeqCollator, SyntheticSeq2Seq
from seq2seq_transformer.embeddings import SinusoidalPositionalEncoding
from seq2seq_transformer.masks import make_causal_mask, make_src_mask, make_tgt_mask
from seq2seq_transformer.scheduler import cosine_lambda, noam_lambda
from seq2seq_transformer.tokenizer import CharTokenizer


def build(cfg, seed=0):
    torch.manual_seed(seed)
    return Transformer(cfg).eval()


def rand_batch(cfg, b=3, s=7, t=6):
    src = torch.randint(4, cfg.src_vocab_size, (b, s))
    tgt = torch.randint(4, cfg.tgt_vocab_size, (b, t))
    return src, tgt


# ---------------------------------------------------------------- attention
def test_sdpa_matches_pytorch_reference():
    q, k, v = (torch.randn(2, 4, 5, 8) for _ in range(3))
    mask = make_causal_mask(5)
    ours, w = scaled_dot_product_attention(q, k, v, mask)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(ours, ref, atol=1e-5)
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 5), atol=1e-5)
    assert torch.all(w.masked_select(~mask.expand_as(w)) == 0)


def test_mha_shapes_and_cross_attention():
    mha = MultiHeadAttention(32, 4).eval()
    x, mem = torch.randn(2, 5, 32), torch.randn(2, 9, 32)
    assert mha(x, x, x).shape == (2, 5, 32)
    mha.store_attn = True
    assert mha(x, mem, mem).shape == (2, 5, 32)
    assert mha.attn_weights.shape == (2, 4, 5, 9)  # (B, heads, Tq, Tk)


def test_masks():
    pad = torch.tensor([[5, 6, 0, 0]])
    assert make_src_mask(pad, 0).tolist() == [[[[True, True, False, False]]]]
    m = make_tgt_mask(pad, 0)[0, 0]
    assert m.shape == (4, 4)
    assert m[3].tolist() == [True, True, False, False]  # causal AND padding
    assert m[0].tolist() == [True, False, False, False]


# ---------------------------------------------------------------- embeddings
def test_positional_encoding_values():
    pe = SinusoidalPositionalEncoding(16, 50).pe[0]
    assert torch.allclose(pe[0, 0::2], torch.zeros(8)) and torch.allclose(pe[0, 1::2], torch.ones(8))
    assert math.isclose(pe[3, 0].item(), math.sin(3), rel_tol=1e-5)
    assert math.isclose(pe[3, 1].item(), math.cos(3), rel_tol=1e-5)


def test_positional_encoding_length_check():
    with pytest.raises(ValueError):
        SinusoidalPositionalEncoding(16, 10)(torch.zeros(1, 11, 16))


# ---------------------------------------------------------------- full model
def test_forward_shape_and_loss_backward(cfg):
    m = build(cfg).train()
    src, tgt = rand_batch(cfg)
    logits = m(src, tgt)
    assert logits.shape == (3, 6, cfg.tgt_vocab_size)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_decoder_is_causal(cfg):
    m = build(cfg)
    src, tgt = rand_batch(cfg, b=1, t=8)
    tgt2 = tgt.clone()
    tgt2[0, 5:] = (tgt2[0, 5:] + 1 - 4) % (cfg.tgt_vocab_size - 4) + 4
    a, b = m(src, tgt), m(src, tgt2)
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-5)
    assert not torch.allclose(a[:, 5:], b[:, 5:], atol=1e-5)


def test_cross_attention_uses_source(cfg):
    m = build(cfg)
    src, tgt = rand_batch(cfg, b=1)
    other = (src - 4 + 1) % (cfg.src_vocab_size - 4) + 4
    assert not torch.allclose(m(src, tgt), m(other, tgt), atol=1e-5)


def test_source_padding_is_ignored(cfg):
    m = build(cfg)
    src, tgt = rand_batch(cfg, b=2, s=5)
    padded = torch.cat([src, torch.zeros(2, 3, dtype=torch.long)], dim=1)
    e1 = m.encode(src, make_src_mask(src, 0))
    e2 = m.encode(padded, make_src_mask(padded, 0))
    assert torch.allclose(e1, e2[:, :5], atol=1e-5)
    assert torch.allclose(m(src, tgt), m(padded, tgt), atol=1e-5)


@pytest.mark.parametrize("ffn", ["swiglu", "relu"])
@pytest.mark.parametrize("norm_first", [False, True])
def test_variants_run(cfg, ffn, norm_first):
    cfg.ffn_type, cfg.norm_first = ffn, norm_first
    m = build(cfg)
    src, tgt = rand_batch(cfg)
    assert torch.isfinite(m(src, tgt)).all()


def test_weight_tying_and_sharing(cfg):
    m = build(cfg)
    assert m.generator.weight is m.tgt_tok.embedding.weight
    assert m.src_tok is not m.tgt_tok
    cfg.share_embeddings = True
    m2 = build(cfg)
    assert m2.src_tok is m2.tgt_tok


def test_default_config_matches_resume_spec():
    c = TransformerConfig()
    assert (c.n_enc_layers, c.n_dec_layers, c.n_heads, c.d_model) == (6, 6, 8, 512)
    assert c.ffn_type == "swiglu"


def test_config_validation(tmp_path):
    with pytest.raises(ValueError):
        TransformerConfig(d_model=100, n_heads=8)
    with pytest.raises(ValueError):
        TransformerConfig(src_vocab_size=10, tgt_vocab_size=12, share_embeddings=True)
    with pytest.raises(ValueError):
        TransformerConfig(ffn_type="gelu")
    c = TransformerConfig(d_model=64, n_heads=4)
    c.save(tmp_path / "c.json")
    assert TransformerConfig.load(tmp_path / "c.json") == c


# ---------------------------------------------------------------- training bits
def test_collator_teacher_forcing_layout():
    batch = [([5, 6, 7], [7, 6, 5]), ([8], [8])]
    out = Seq2SeqCollator()(batch)
    assert out["src"].tolist() == [[5, 6, 7, 2], [8, 2, 0, 0]]
    assert out["tgt_in"].tolist() == [[1, 7, 6, 5], [1, 8, 0, 0]]
    assert out["tgt_out"].tolist() == [[7, 6, 5, 2], [8, 2, 0, 0]]


def test_synthetic_tasks():
    for task, fn in [("copy", lambda s: s), ("reverse", lambda s: s[::-1]), ("sort", sorted)]:
        ds = SyntheticSeq2Seq(task, 20, 30, 3, 9, seed=1)
        for s, t in ds.items:
            assert t == fn(s) and all(4 <= x < 30 for x in s)


def test_noam_schedule_peaks_at_warmup():
    f = noam_lambda(512, 100)
    vals = [f(i) for i in range(300)]
    assert max(vals) == vals[99]
    assert vals[0] < vals[50] < vals[99] and vals[150] > vals[250]
    assert math.isclose(vals[99], 512 ** -0.5 * 100 ** -0.5, rel_tol=1e-6)


def test_cosine_schedule():
    f = cosine_lambda(10, 100, 0.1)
    assert f(0) < f(9) <= 1.0
    assert math.isclose(f(10), 1.0, rel_tol=1e-6)
    assert math.isclose(f(100), 0.1, rel_tol=1e-6)


def test_overfits_tiny_batch(cfg):
    torch.manual_seed(0)
    m = Transformer(cfg).train()
    opt = m.configure_optimizer(lr=2e-3)
    src = torch.randint(4, cfg.src_vocab_size, (8, 6))
    bos = torch.full((8, 1), cfg.bos_id)
    eos = torch.full((8, 1), cfg.eos_id)
    tin, tout = torch.cat([bos, src], 1), torch.cat([src, eos], 1)
    first = last = None
    for _ in range(150):
        loss = F.cross_entropy(m(src, tin).view(-1, cfg.tgt_vocab_size), tout.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < 0.3 * first


def test_greedy_decode(cfg):
    m = build(cfg)
    src, _ = rand_batch(cfg, b=2)
    out = greedy_decode(m, src, max_len=10)
    assert out.shape[0] == 2 and out.shape[1] <= 10
    assert out.min() >= 0 and out.max() < cfg.tgt_vocab_size
    assert torch.equal(out, greedy_decode(m, src, max_len=10))  # deterministic


def test_tokenizer_roundtrip():
    tok = CharTokenizer.from_texts(["héllo wörld", "abc"])
    assert tok.decode(tok.encode("héllo")) == "héllo"
    assert tok.encode("§")[0] == 3  # UNK
