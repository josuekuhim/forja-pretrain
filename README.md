# forja-pretrain

**A decoder-only Transformer from scratch, pretrained on Portuguese + code with the tokenizer of [Forja](https://github.com/josuekuhim/forja)'s base model.**

> Result: _pending_. Validation perplexity, tokens/s and the acceptance rate as a draft model go here once it is trained.

## Role in the system

It uses **the same tokenizer as the base model** (Qwen2.5-Coder, see [ADR 0001](https://github.com/josuekuhim/forja/blob/main/docs/adr/0001-modelo-base.md)). That is why it becomes the *draft model* for speculative decoding in `forja-serve`.

## Architecture

| Component | Implementation | File |
| --- | --- | --- |
| Normalization | RMSNorm (pre-norm, in fp32) | `model.py` |
| Position | RoPE (half/half, like Llama/Qwen) | `model.py` |
| Attention | causal, MHA / GQA / MQA via `n_kv_heads` | `model.py` |
| FFN | SwiGLU, hidden ≈ 8/3·d | `model.py` |
| Output | `lm_head` tied to the embedding | `model.py` |
| Training | AdamW (no decay on 1D params), warmup + cosine/WSD, grad accum, AMP, clipping | `train.py` |

### Why 69M parameters if the target is 10–50M?

Qwen's vocabulary has ~152k tokens. With `d_model=384`, the embedding alone has 58M. This repo reports **non-embedding parameters** (the Kaplan/Chinchilla convention):

| Config | Non-embedding | Total |
| --- | --- | --- |
| `base` (d384, 6 layers) | 10.6M | 69.0M |
| d384, 12 layers | 21.2M | 79.6M |
| d512, 8 layers | 25.7M | 103.5M |

It is the price of reusing the tokenizer for speculative decoding, and it is documented as a limitation.

## How to reproduce

```bash
uv sync                      # local CPU
uv run pytest                # shapes, causal mask, gradients, RoPE, overfitting, training smoke test

# Data (on a GPU/Kaggle/Colab, or locally with patience)
uv sync --extra data
uv run python scripts/prepare_data.py configs/data.yaml   # writes data/{train,val}.bin + manifest.json

# Training
uv run forja-train configs/base.yaml
uv run forja-train configs/base.yaml model.n_kv_heads=2 train.out_dir=runs/gqa2   # ablation
```

Each run saves to `runs/<name>/`: resolved `config.yaml`, `history.jsonl`, `summary.json`, `best.pt` and `last.pt`.

### On a free GPU (Kaggle or Colab)

Open [`notebooks/kaggle_pretrain.ipynb`](notebooks/kaggle_pretrain.ipynb), turn on the T4 GPU and run everything. It clones the repo, prepares the corpus, trains with [`configs/kaggle-t4.yaml`](configs/kaggle-t4.yaml), generates the charts and exports (or publishes to the Hub) in the contract's format.

- **Resuming:** a full checkpoint (`last.pt`: model, optimizer, scaler and the sampler's RNG state) every 250 steps, written atomically. Running again resumes from the same step, with a result **identical** to continuous training (`test_retomada_identica_ao_treino_continuo`).
- **Session limit:** `train.max_hours: 11` saves and exits before Kaggle kills the session at 12 h.
- **Memory:** with a 152k vocab, the logits dominate VRAM (8 × 512 × 151936 in fp32 = 2.5 GB just for the loss). The T4 config uses micro-batch 8 with accum 8, keeping 32k tokens per step.

```bash
uv run python scripts/plot_history.py runs/a runs/b --out comparacao.png   # curves from several runs
uv run python scripts/export_hub.py runs/a export/a [--push user/repo]     # safetensors + model card
```

## Tests

- `test_mascara_causal`: changing future tokens does not change past logits (MHA and MQA).
- `test_gradientes_chegam_em_todos_os_parametros`: gradients reach every parameter.
- `test_rope_...`: preserves the norm, and the dot product depends only on relative distance.
- `test_loss_inicial_perto_de_ln_vocab`: sane initialization.
- `test_overfit_um_batch` and `test_smoke_treino_aprende_padrao`: the loop learns.

## Planned experiments (one variable at a time)

| # | Variable | Values |
| --- | --- | --- |
| E1 | Attention | MHA (6 kv) · GQA (2 kv) · MQA (1 kv) |
| E2 | Context | 256 · 512 · 1024 |
| E3 | Schedule | cosine · WSD · constant |
| E4 | Depth | 6 · 12 layers |
| E5 | Draft | acceptance rate vs. Qwen2.5-Coder (anticipates forja-serve) |

## Limitations

- Small model and corpus; it is not a useful model on its own, it is a draft and an object of study.
- Exact dedup only (hash); near-dup with MinHash is left for v2.
- `generate` without a KV cache, on purpose: the KV cache is implemented in `forja-serve`.
