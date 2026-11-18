"""Loop de treino reproduzível.

Uso:
    uv run forja-train configs/base.yaml
    uv run forja-train configs/base.yaml train.max_steps=200 model.n_kv_heads=2
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

from forja_pretrain.config import ExperimentConfig, load_config
from forja_pretrain.data import TokenStream
from forja_pretrain.model import Transformer
from forja_pretrain.schedule import lr_at


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_optimizer(model: torch.nn.Module, cfg: ExperimentConfig, device: str) -> torch.optim.Optimizer:
    # Sem weight decay em normas e vetores 1D (prática padrão).
    decay, no_decay = [], []
    seen = set()
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.train.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.train.lr, betas=cfg.train.betas, fused=device == "cuda")


@torch.no_grad()
def evaluate(model, stream: TokenStream, n_batches: int, batch_size: int, device: str, autocast) -> float:
    model.eval()
    g = torch.Generator().manual_seed(0)  # mesmas janelas de validação em todo eval
    losses = []
    for _ in range(n_batches):
        x, y = stream.sample(batch_size, g, device)
        with autocast:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train(cfg: ExperimentConfig) -> dict:
    tc = cfg.train
    seed_everything(tc.seed)
    device = pick_device()
    out = Path(tc.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[tc.dtype]
    if device == "cpu":
        dtype = torch.float32 if tc.dtype == "float16" else dtype
    autocast = torch.autocast(device_type=device, dtype=dtype) if dtype != torch.float32 else nullcontext()
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16 and device == "cuda"))

    train_s = TokenStream(cfg.data.train_bin, cfg.data.seq_len)
    val_s = TokenStream(cfg.data.val_bin, cfg.data.seq_len)

    model = Transformer(cfg.model).to(device)
    print(
        f"[forja] device={device} params(não-embedding)={model.num_params() / 1e6:.2f}M "
        f"params(total)={model.num_params(False) / 1e6:.2f}M"
    )
    opt = build_optimizer(model, cfg, device)
    fwd = torch.compile(model) if tc.compile else model

    wandb_run = None
    if tc.wandb_project:
        import wandb

        wandb_run = wandb.init(project=tc.wandb_project, name=cfg.name, config=cfg.to_dict())

    g = torch.Generator().manual_seed(tc.seed)
    tokens_per_step = tc.batch_size * tc.grad_accum * cfg.data.seq_len
    history: list[dict] = []
    best_val = math.inf
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for step in range(tc.max_steps + 1):
        lr = lr_at(
            step,
            lr=tc.lr,
            warmup=tc.warmup_steps,
            max_steps=tc.max_steps,
            min_ratio=tc.min_lr_ratio,
            kind=tc.schedule,
        )
        for group in opt.param_groups:
            group["lr"] = lr

        if step % tc.eval_every == 0 or step == tc.max_steps:
            val_loss = evaluate(fwd, val_s, tc.eval_batches, tc.batch_size, device, autocast)
            rec = {"step": step, "val_loss": val_loss, "val_ppl": math.exp(min(val_loss, 20))}
            history.append(rec)
            print(f"[eval] step={step} val_loss={val_loss:.4f} val_ppl={rec['val_ppl']:.2f}")
            if wandb_run:
                wandb_run.log(rec, step=step)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {"model": model.state_dict(), "config": cfg.to_dict(), "step": step}, out / "best.pt"
                )
        if step == tc.max_steps:
            break

        loss_acc = 0.0
        for _ in range(tc.grad_accum):
            x, y = train_s.sample(tc.batch_size, g, device)
            with autocast:
                _, loss = fwd(x, y)
            loss = loss / tc.grad_accum
            loss_acc += loss.item()
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        if step % tc.log_every == 0:
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            t0 = time.perf_counter()
            steps_done = tc.log_every if step > 0 else 1
            rec = {
                "step": step,
                "train_loss": loss_acc,
                "lr": lr,
                "grad_norm": grad_norm,
                "tok_per_s": tokens_per_step * steps_done / dt,
            }
            history.append(rec)
            print(
                f"[train] step={step} loss={loss_acc:.4f} lr={lr:.2e} gnorm={grad_norm:.2f} "
                f"tok/s={rec['tok_per_s']:.0f}"
            )
            if wandb_run:
                wandb_run.log(rec, step=step)

    summary = {
        "name": cfg.name,
        "best_val_loss": best_val,
        "best_val_ppl": math.exp(min(best_val, 20)),
        "params_non_embedding": model.num_params(),
        "params_total": model.num_params(False),
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
        "device": device,
        "torch": torch.__version__,
    }
    (out / "history.jsonl").write_text("\n".join(json.dumps(h) for h in history) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if wandb_run:
        wandb_run.summary.update(summary)
        wandb_run.finish()
    return summary


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cfg = load_config(sys.argv[1], sys.argv[2:])
    print(json.dumps(train(cfg), indent=2))


if __name__ == "__main__":
    main()
