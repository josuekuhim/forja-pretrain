import math

import pytest
import torch

from forja_pretrain.config import ModelConfig
from forja_pretrain.model import RMSNorm, Transformer, apply_rope, rope_cache


def tiny(**kw) -> ModelConfig:
    base = dict(vocab_size=97, d_model=32, n_layers=2, n_heads=4, n_kv_heads=4, max_seq_len=16)
    return ModelConfig(**{**base, **kw})


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.mark.parametrize("n_kv_heads", [4, 2, 1])  # MHA, GQA, MQA
def test_shapes(n_kv_heads):
    m = Transformer(tiny(n_kv_heads=n_kv_heads))
    x = torch.randint(0, 97, (3, 10))
    logits, loss = m(x, x)
    assert logits.shape == (3, 10, 97)
    assert loss.ndim == 0


def test_loss_inicial_perto_de_ln_vocab():
    m = Transformer(tiny())
    x = torch.randint(0, 97, (8, 16))
    y = torch.randint(0, 97, (8, 16))  # targets != input: com embedding amarrado, o próprio token ganha logit
    _, loss = m(x, y)
    assert abs(loss.item() - math.log(97)) < 0.5


@pytest.mark.parametrize("n_kv_heads", [4, 1])
def test_mascara_causal(n_kv_heads):
    """Mudar tokens futuros não pode alterar os logits das posições anteriores."""
    m = Transformer(tiny(n_kv_heads=n_kv_heads)).eval()
    x = torch.randint(0, 97, (1, 12))
    x2 = x.clone()
    x2[0, 7:] = (x2[0, 7:] + 1) % 97
    a, _ = m(x)
    b, _ = m(x2)
    torch.testing.assert_close(a[0, :7], b[0, :7])
    assert not torch.allclose(a[0, 7:], b[0, 7:])


def test_gradientes_chegam_em_todos_os_parametros():
    m = Transformer(tiny(n_kv_heads=2))
    x = torch.randint(0, 97, (2, 16))
    _, loss = m(x, x)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_embedding_amarrado_e_contagem_de_parametros():
    m = Transformer(tiny())
    assert m.lm_head.weight.data_ptr() == m.embed.weight.data_ptr()
    assert m.num_params(non_embedding=False) - m.num_params() == 97 * 32
    untied = Transformer(tiny(tie_embeddings=False))
    assert untied.num_params(False) - untied.num_params() == 2 * 97 * 32


def test_rope_preserva_norma_e_codifica_posicao_relativa():
    cos, sin = rope_cache(16, 8, 10_000.0)
    q = torch.randn(1, 1, 1, 8).expand(1, 1, 16, 8)
    k = torch.randn(1, 1, 1, 8).expand(1, 1, 16, 8)
    rq, rk = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    torch.testing.assert_close(rq.norm(dim=-1), q.norm(dim=-1))
    # <q_m, k_n> depende só de m - n.
    s = rq[0, 0] @ rk[0, 0].T
    torch.testing.assert_close(s[3, 1], s[10, 8], atol=1e-5, rtol=1e-5)


def test_rmsnorm():
    x = torch.randn(4, 32) * 10
    y = RMSNorm(32)(x)
    torch.testing.assert_close(y.pow(2).mean(-1), torch.ones(4), atol=1e-4, rtol=1e-4)


def test_overfit_um_batch():
    """Se não decora um batch, tem bug no modelo ou no loop."""
    m = Transformer(tiny())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 97, (4, 16))
    for _ in range(150):
        _, loss = m(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_generate_greedy_deterministico():
    m = Transformer(tiny()).eval()
    x = torch.randint(0, 97, (1, 4))
    a = m.generate(x, 6, temperature=0)
    b = m.generate(x, 6, temperature=0)
    assert a.shape == (1, 10)
    assert torch.equal(a, b)


def test_config_invalida():
    with pytest.raises(ValueError):
        ModelConfig(d_model=30, n_heads=4)
    with pytest.raises(ValueError):
        ModelConfig(n_heads=6, n_kv_heads=4)
