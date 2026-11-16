import pytest

from seq2seq_transformer import TransformerConfig


@pytest.fixture
def cfg():
    return TransformerConfig(
        src_vocab_size=20, tgt_vocab_size=20, d_model=32, n_heads=4,
        n_enc_layers=2, n_dec_layers=2, d_ff=64, dropout=0.0, max_len=32,
    )
