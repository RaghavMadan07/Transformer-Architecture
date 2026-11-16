import pytest

from modern_transformer import TransformerConfig


@pytest.fixture
def cfg():
    return TransformerConfig(
        vocab_size=100, max_seq_len=64, d_model=64, n_layers=3, n_heads=4, n_kv_heads=2
    )


@pytest.fixture
def moe_cfg():
    return TransformerConfig(
        vocab_size=100, max_seq_len=64, d_model=64, n_layers=4, n_heads=4, n_kv_heads=2,
        use_moe=True, moe_every=2, n_experts=4, moe_top_k=2,
    )
