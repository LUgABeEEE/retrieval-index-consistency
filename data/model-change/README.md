# Encoder-swap panel (2026-11-19)

The paper's model-change panel originally used a quantised local implementation whose vectors sit at
cosine 0.88 from the store's. That is close enough that the swap looks like noise rather than a new
space, so it did not exercise the case the paper claims to handle.

This directory holds a clean repeat with two genuinely independent encoders:

- store: `bge-m3` fp16 CLS, 131,028 x 1024
- new model: `BAAI/bge-large-zh-v1.5` (BERT-large, 326M), 1024-dim
- cross-model per-row cosine: **0.0254** (effectively unrelated)

| Arm | recall@5 | PER@10 | stale rows |
|---|---|---|---|
| naive (old rows kept) | 0.4312 | 45.05% | 131,028 |
| protocol (identity-filtered) | 0.4954 | 0.00% | 0 |
| full re-embed, new model | 0.5229 | 0.00% | 0 |

Reproduce: `PYTHONPATH=../hpc/pylibs python3 /tmp/mc2.py` (script archived alongside the results).
