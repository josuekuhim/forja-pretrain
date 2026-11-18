from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 151_936  # linhas do embedding do Qwen2.5 (ver ADR 0001 no repo forja)
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 6  # < n_heads ativa GQA; 1 = MQA
    ffn_hidden: int | None = None  # None -> ~8/3 * d_model, múltiplo de 64 (SwiGLU)
    max_seq_len: int = 512
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model precisa ser divisível por n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads precisa ser divisível por n_kv_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("head_dim precisa ser par para o RoPE")
        if self.ffn_hidden is None:
            self.ffn_hidden = 64 * ((int(8 * self.d_model / 3) + 63) // 64)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class DataConfig:
    train_bin: str = "data/train.bin"
    val_bin: str = "data/val.bin"
    seq_len: int = 512


@dataclass
class TrainConfig:
    seed: int = 1337
    batch_size: int = 16
    grad_accum: int = 4
    max_steps: int = 5_000
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    schedule: Literal["cosine", "constant", "wsd"] = "cosine"
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 50
    log_every: int = 10
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    compile: bool = False
    out_dir: str = "runs/default"
    wandb_project: str | None = None
    # Sessões de GPU gratuitas caem (Kaggle: 12 h). Checkpoint completo a cada N passos e retomada automática.
    ckpt_every: int = 250
    resume: bool = True
    max_hours: float | None = None  # salva e encerra antes do limite da sessão
    stop_at_step: int | None = None  # para no passo N (sessões curtas, testes de retomada)


@dataclass
class ExperimentConfig:
    name: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type, raw: dict[str, Any] | None):
    raw = raw or {}
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Chaves desconhecidas em {cls.__name__}: {sorted(unknown)}")
    if "betas" in raw:
        raw = {**raw, "betas": tuple(raw["betas"])}
    return cls(**raw)


def _parse_scalar(value: str) -> Any:
    parsed = yaml.safe_load(value)
    if isinstance(parsed, str):  # PyYAML (YAML 1.1) lê "1e-4" como string
        try:
            return float(parsed)
        except ValueError:
            return parsed
    return parsed


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Carrega YAML e aplica overrides no formato `secao.chave=valor` (valor em YAML)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for ov in overrides or []:
        key, _, value = ov.partition("=")
        node = raw
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _parse_scalar(value)
    return ExperimentConfig(
        name=raw.get("name", Path(path).stem),
        model=_build(ModelConfig, raw.get("model")),
        data=_build(DataConfig, raw.get("data")),
        train=_build(TrainConfig, raw.get("train")),
    )
