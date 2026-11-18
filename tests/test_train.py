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


def _periodic_data(tmp_path):
    rng = np.random.default_rng(0)
    data = np.tile(rng.integers(0, 256, size=37), 2000)
    write_tokens(tmp_path / "train.bin", data)
    write_tokens(tmp_path / "val.bin", data[:5000])
    return [
        f"data.train_bin={(tmp_path / 'train.bin').as_posix()}",
        f"data.val_bin={(tmp_path / 'val.bin').as_posix()}",
    ]


def test_retomada_identica_ao_treino_continuo(tmp_path):
    """Interromper no passo 25 e retomar tem que dar exatamente o mesmo resultado que treinar direto."""
    data = _periodic_data(tmp_path)
    corrido = train(
        load_config("configs/smoke.yaml", [*data, f"train.out_dir={(tmp_path / 'a').as_posix()}"])
    )

    b = [*data, f"train.out_dir={(tmp_path / 'b').as_posix()}"]
    parcial = train(load_config("configs/smoke.yaml", [*b, "train.stop_at_step=25"]))
    assert not parcial["concluido"] and parcial["passos"] == 25
    retomado = train(load_config("configs/smoke.yaml", b))

    assert retomado["concluido"]
    assert retomado["best_val_loss"] == pytest.approx(corrido["best_val_loss"], abs=1e-6)
    hist = [json.loads(line) for line in (tmp_path / "b" / "history.jsonl").read_text().splitlines()]
    assert [h["step"] for h in hist if "val_loss" in h] == [0, 30, 60]  # sem eval duplicado


def test_retomada_recusa_arquitetura_diferente(tmp_path):
    data = _periodic_data(tmp_path)
    b = [*data, f"train.out_dir={(tmp_path / 'b').as_posix()}"]
    train(load_config("configs/smoke.yaml", [*b, "train.stop_at_step=10"]))
    with pytest.raises(ValueError, match="outra arquitetura"):
        train(load_config("configs/smoke.yaml", [*b, "model.n_layers=3"]))
