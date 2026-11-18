import json

import numpy as np
import pytest

from forja_pretrain.config import load_config
from forja_pretrain.data import write_tokens
from forja_pretrain.schedule import lr_at
from forja_pretrain.train import train


def test_schedule():
    kw = dict(lr=1.0, warmup=10, max_steps=100, min_ratio=0.1)
    assert lr_at(0, kind="cosine", **kw) == pytest.approx(0.1)
    assert lr_at(10, kind="cosine", **kw) == pytest.approx(1.0)
    assert lr_at(100, kind="cosine", **kw) == pytest.approx(0.1)
    assert lr_at(50, kind="wsd", **kw) == pytest.approx(1.0)
    assert lr_at(100, kind="wsd", **kw) == pytest.approx(0.1)


def test_overrides(tmp_path):
    cfg = load_config("configs/smoke.yaml", ["model.n_kv_heads=1", "train.lr=1e-4"])
    assert cfg.model.n_kv_heads == 1
    assert cfg.train.lr == pytest.approx(1e-4)
    with pytest.raises(ValueError):
        load_config("configs/smoke.yaml", ["model.nao_existe=1"])


def test_smoke_treino_aprende_padrao(tmp_path):
    """Sequência periódica: o modelo tem que derrubar a loss bem abaixo de ln(vocab)."""
    rng = np.random.default_rng(0)
    pattern = rng.integers(0, 256, size=37)
    data = np.tile(pattern, 2000)
    write_tokens(tmp_path / "train.bin", data)
    write_tokens(tmp_path / "val.bin", data[:5000])

    cfg = load_config(
        "configs/smoke.yaml",
        [
            f"data.train_bin={(tmp_path / 'train.bin').as_posix()}",
            f"data.val_bin={(tmp_path / 'val.bin').as_posix()}",
            f"train.out_dir={(tmp_path / 'run').as_posix()}",
        ],
    )
    summary = train(cfg)
    assert summary["best_val_loss"] < 1.0  # ln(256) ≈ 5.5
    assert (tmp_path / "run" / "best.pt").exists()
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["name"] == "smoke"
