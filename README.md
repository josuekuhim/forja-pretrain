# forja-pretrain

**Transformer decoder-only do zero, pré-treinado em português + código com o tokenizer do modelo base do [Forja](../forja).**

> Resultado: _pendente_ — val ppl, tokens/s e taxa de aceitação como draft model entram aqui.

## Papel no sistema

Usa **o mesmo tokenizer do modelo base** (Qwen2.5-Coder, ver [ADR 0001](../forja/docs/adr/0001-modelo-base.md)). Por isso vira o *draft model* do speculative decoding no `forja-serve`.

## Arquitetura

| Componente | Implementação | Arquivo |
| --- | --- | --- |
| Normalização | RMSNorm (pre-norm, em fp32) | `model.py` |
| Posição | RoPE (metade/metade, igual ao Llama/Qwen) | `model.py` |
| Atenção | causal, MHA / GQA / MQA via `n_kv_heads` | `model.py` |
| FFN | SwiGLU, hidden ≈ 8/3·d | `model.py` |
| Saída | `lm_head` amarrado ao embedding | `model.py` |
| Treino | AdamW (sem decay em 1D), warmup + cosine/WSD, grad accum, AMP, clip | `train.py` |

### Por que 69M parâmetros se o alvo é 10–50M?

O vocabulário do Qwen tem ~152k tokens. Com `d_model=384`, só o embedding tem 58M. Este repo reporta **parâmetros não-embedding** (convenção de Kaplan/Chinchilla):

| Config | Não-embedding | Total |
| --- | --- | --- |
| `base` (d384, 6 camadas) | 10,6M | 69,0M |
| d384, 12 camadas | 21,2M | 79,6M |
| d512, 8 camadas | 25,7M | 103,5M |

É o preço de reaproveitar o tokenizer para o speculative decoding, e está documentado como limitação.

## Como reproduzir

```bash
uv sync                      # CPU local
uv run pytest                # formas, máscara causal, gradientes, RoPE, overfit, smoke de treino

# Dados (em GPU/Kaggle/Colab, ou local com paciência)
uv sync --extra data
uv run python scripts/prepare_data.py configs/data.yaml   # gera data/{train,val}.bin + manifest.json

# Treino
uv run forja-train configs/base.yaml
uv run forja-train configs/base.yaml model.n_kv_heads=2 train.out_dir=runs/gqa2   # ablação
```

Cada execução salva em `runs/<nome>/`: `config.yaml` resolvida, `history.jsonl`, `summary.json` e `best.pt`.

## Testes

- `test_mascara_causal`: mudar tokens futuros não altera logits passados (MHA e MQA).
- `test_gradientes_chegam_em_todos_os_parametros`.
- `test_rope_...`: preserva norma e o produto escalar depende só da distância relativa.
- `test_loss_inicial_perto_de_ln_vocab`: inicialização sã.
- `test_overfit_um_batch` e `test_smoke_treino_aprende_padrao`: o loop aprende.

## Experimentos planejados (uma variável por vez)

| # | Variável | Valores |
| --- | --- | --- |
| E1 | Atenção | MHA (6 kv) · GQA (2 kv) · MQA (1 kv) |
| E2 | Contexto | 256 · 512 · 1024 |
| E3 | Schedule | cosine · WSD · constante |
| E4 | Profundidade | 6 · 12 camadas |
| E5 | Draft | taxa de aceitação vs. Qwen2.5-Coder (antecipa o forja-serve) |

## Limitações

- Modelo e corpus pequenos; não é um modelo útil sozinho, é um draft e um objeto de estudo.
- Dedup apenas exata (hash); near-dup com MinHash fica para a v2.
- `generate` sem KV cache, de propósito: o KV cache é implementado no `forja-serve`.
