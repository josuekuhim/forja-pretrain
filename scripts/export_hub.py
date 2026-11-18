"""Exporta um checkpoint para o formato do contrato do Forja: safetensors + config.json + model card.

Uso:
    python scripts/export_hub.py runs/<nome> export/<nome> [--tokenizer Qwen/Qwen2.5-Coder-1.5B-Instruct]
    python scripts/export_hub.py runs/<nome> export/<nome> --push usuario/forja-pretrain-<nome>   # precisa de HF_TOKEN
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forja_pretrain.config import ModelConfig  # noqa: E402
from forja_pretrain.model import Transformer  # noqa: E402

CARD = """---
license: apache-2.0
language: [pt, code]
tags: [forja, pretraining, draft-model]
---

# {name}

Transformer decoder-only de **{non_emb:.1f}M parâmetros não-embedding** ({total:.1f}M no total), treinado do zero
no projeto [Forja](https://github.com) como *draft model* para speculative decoding. Usa o tokenizer
`{tokenizer}`, então os IDs de token batem com o modelo base.

| Métrica | Valor |
| --- | --- |
| Melhor val loss | {val_loss:.4f} |
| Perplexidade de validação | {val_ppl:.2f} |
| Passos | {passos} |
| Treino concluído | {concluido} |

Arquitetura: RMSNorm, RoPE, atenção causal (n_kv_heads={n_kv}), SwiGLU, embedding amarrado ao `lm_head`.
Código, config e histórico: repositório `forja-pretrain` (este model card é gerado por `scripts/export_hub.py`).

**Limitações:** modelo pequeno e corpus pequeno; não é útil sozinho. Serve como draft e como objeto de estudo.
"""


def export(run_dir: Path, out_dir: Path, tokenizer: str) -> Path:
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**ck["config"]["model"])
    model = Transformer(mcfg)
    model.load_state_dict(ck["model"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model(
        model, str(out_dir / "model.safetensors")
    )  # lida com o peso compartilhado (embedding amarrado)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    (out_dir / "config.json").write_text(
        json.dumps(
            {"architecture": "forja-pretrain", "tokenizer": tokenizer, **ck["config"]["model"]}, indent=2
        ),
        encoding="utf-8",
    )
    for f in ("summary.json", "history.jsonl", "config.yaml"):
        if (run_dir / f).exists():
            (out_dir / f).write_bytes((run_dir / f).read_bytes())
    (out_dir / "README.md").write_text(
        CARD.format(
            name=summary["name"],
            non_emb=summary["params_non_embedding"] / 1e6,
            total=summary["params_total"] / 1e6,
            tokenizer=tokenizer,
            val_loss=summary["best_val_loss"],
            val_ppl=summary["best_val_ppl"],
            passos=summary.get("passos", "?"),
            concluido="sim" if summary.get("concluido", True) else "não (retomável)",
            n_kv=mcfg.n_kv_heads,
        ),
        encoding="utf-8",
    )
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--push", metavar="REPO_ID", help="Publica no Hugging Face Hub (usa HF_TOKEN)")
    a = ap.parse_args()
    out = export(a.run_dir, a.out_dir, a.tokenizer)
    print(f"exportado em {out}")
    if a.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(a.push, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=a.push)
        print(f"publicado em https://huggingface.co/{a.push}")


if __name__ == "__main__":
    main()
