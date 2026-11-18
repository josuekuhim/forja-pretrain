"""Gráficos de loss/perplexidade a partir de runs/<nome>/history.jsonl (um ou vários runs, para comparar).

Uso: python scripts/plot_history.py runs/a [runs/b ...] --out loss.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(run: Path) -> tuple[list[dict], list[dict]]:
    rows = [
        json.loads(line) for line in (run / "history.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
    return [r for r in rows if "train_loss" in r], [r for r in rows if "val_loss" in r]


def plot(runs: list[Path], out: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in runs:
        tr, va = load(run)
        line = ax1.plot(
            [r["step"] for r in tr], [r["train_loss"] for r in tr], alpha=0.5, label=f"{run.name} treino"
        )
        ax1.plot(
            [r["step"] for r in va],
            [r["val_loss"] for r in va],
            "o-",
            color=line[0].get_color(),
            label=f"{run.name} validação",
        )
        ax2.plot(
            [r["step"] for r in va], [math.exp(min(r["val_loss"], 20)) for r in va], "o-", label=run.name
        )
    ax1.set(xlabel="passo", ylabel="loss (nats/token)", title="Loss")
    ax2.set(xlabel="passo", ylabel="perplexidade", title="Perplexidade de validação", yscale="log")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("loss.png"))
    a = ap.parse_args()
    print(plot(a.runs, a.out))
