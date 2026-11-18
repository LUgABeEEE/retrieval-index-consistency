# Retrieval Index Consistency — Artifact

Reproduction artifact for the manuscript:

> **Budget-Constrained Index Repair in an Evolving Retrieval Corpus: Measurement,
> Protocol, and Scheduling**
> Qidong Gao — Artificial Intelligence Joint Institute of UD and ECUT,
> East China University of Technology

This repository contains the **data and verification scripts** behind every number in the
paper. The manuscript itself is not hosted here while it is under review.

## What is in here

```
data/      19 CSV/JSON files  — the aggregates behind every reported number
scripts/   three standalone Python scripts (standard library only, plus pypdf)
protocols/ four pre-registered protocols, frozen before the runs they govern
figures/   seven vector SVG figures, generated from data by script
```

### Protocols

`protocols/` holds the pre-registration documents. Each was written before the experiment it
governs, and each records the criteria that were then applied unchanged. They are plain Markdown
so that their text can be diffed and quoted.

| File | Governs |
|---|---|
| `e1-PROTOCOL.md` | The five update events and their severity criteria |
| `e3-PROTOCOL.md` | The scheduling comparison, Section 9 records the one criterion change |
| `chunk-sweep-PROTOCOL.md` | The chunk-size sweep (four criteria fixed before the run) |
| `density-sweep-PROTOCOL.md` | The controlled density experiment of Appendix D |
| `online-bandit-PROTOCOL.md` | The online variant (frozen 11 Nov 20:50, before its formal run) |
| `reps20-PROTOCOL.md` | The 20-repetition resplit |
| `rule2-ablation-PROTOCOL.md` | The membership-check ablation |
| `rule3-ablation-PROTOCOL.md` | The cache-key ablation |

Seven of the eight protocols have a sibling `*-protocol.sha256`. There is no hash for E3: that
protocol was edited after a dry run, and the manuscript discloses this in Section 5 rather than
papering over it.

**What the digests cover.** They are digests of the *published* text, which differs from the frozen
original in one respect: production paths appear as `$KB_ROOT` and `$REPO_ROOT`. For most protocols
the two texts are otherwise identical. For E1 and the chunk sweep they are not, because those digests
were first computed over the unsanitised original, which is not published. Rather than leave two
digests that no reader could verify, both have been recomputed over the published text. What a reader
can therefore check is that the file has not changed since publication; what cannot be checked, for
those two, is that the published text matches the text frozen before the run.

`experiments/` holds the scripts that produce the scheduling results, so that the per-run values in
`data/results.json` can be regenerated on a corpus of the reader's own. Two subdirectories carry
their own pre-registrations: `experiments/e3/reps20/` (the 20-repetition resplit) and
`experiments/e3/online-bandit/` (the online variant, frozen 11 Nov 20:50 before its formal run). They expect the
paths in `$REPO_ROOT` and `$KB_ROOT` to be populated; on a different corpus, start from the
`chunk-sweep` protocol, which is the cheapest way to check that the pipeline reproduces.

These documents use `$KB_ROOT` and `$REPO_ROOT` as placeholders where the production paths were.
That is the only edit made to them for publication; no criterion or threshold was altered.

### Data files

| File | Contents |
|---|---|
| `table1-e1-events.csv` | E1: five update classes × two magnitudes — SVR, PER, RDU |
| `table2-e2-four-arms.csv` | E2: naive upsert / prune-only / protocol / full rebuild |
| `table3-e3-scheduling.csv` | E3: six strategies × three budgets × three scenarios |
| `table-beir.csv` | External replication, three BEIR subsets |
| `table-cmteb.csv` | External replication, two C-MTEB subsets |
| `table-rule2.csv` | Ablation: remove the pre-load membership check |
| `table-rule3.csv` | Ablation: wrong cache key |
| `ext_all_5060.csv` | Five corpora compared under one harness (TREC-COVID, scidocs, FiQA, arguana, KB) |
| `table_5060.csv` | Controlled density sweep, seven tiers (k = 1 … 16) |
| `density_5060.json` | Raw output of the density sweep |
| `base_5060.json` | Cross-platform baseline re-run (RTX 5060, transformers) |
| `production-audit-20261106.csv` | The production audit behind the 96.1% orphan figure |
| `facts.json` | 293 registered numbers: value, paper precision, source field |

### Scripts

| Script | Purpose |
|---|---|
| `check_numbers.py` | Verifies the manuscript is internally consistent: no placeholders, 293 facts agree between language versions, 38 tables byte-identical to their generators, every `##` heading present in the rendered PDF |
| `verify.py` | Samples numbers from the manuscript and locates their source in `data/` |
| `bm25vec.py` | Vectorised BM25 scoring (148× faster than the per-posting loop; numerically equivalent) |

## How to check the numbers

```bash
python3 scripts/check_numbers.py        # expects PASS
python3 scripts/verify.py --sample 20   # traces 20 random numbers back to data/
```

`verify.py` needs the manuscript to sample from. Point it at your copy:

```bash
python3 scripts/verify.py --paper /path/to/paper-v0.1.md --sample 20
```

`check_numbers.py` additionally needs the 38 table generators and both language versions
side by side, so it is shipped here for inspection rather than for standalone use.

## What the numbers say

Three claims are worth checking first, because the rest of the paper rests on them:

1. **The failure is large and silent.** One re-chunking left 75,225 of 78,313 vector rows
   (96.1%) without a matching chunk, while those rows kept being loaded and fused into
   rankings. Source: `production-audit-20261106.csv`.

2. **Recall rewards a broken system.** Naive index maintenance *raises* recall@5
   (0.5138 vs 0.5046) while leaving 19.72% of top-10 evidence provenance-wrong. Source:
   `table2-e2-four-arms.csv`.

3. **The protocol makes this class of failure structurally impossible.** The protocol arm holds
   PER at 0.00% across five external corpora (14 cells), with document-level PER exactly zero in
   all ten measurements, at 0.0125% of full-rebuild cost in steady state. Source:
   `ext_all_5060.csv`, `table2-e2-four-arms.csv`.

## What the paper does not explain

The absolute cross-corpus PER varies from 1.20% to 19.72% and we could not explain the spread.
Three candidate mechanisms were tested and all failed:

- **Block density** — a controlled sweep from k = 1 to k = 16 gives Spearman rho = −0.926
  (p = 0.0049), i.e. the opposite direction. See `table_5060.csv`.
- **Stale-vector rate** — the relation PER ≈ 1.05 × SVR holds only at fixed density; across
  densities the same SVR of 0.100 gives 5.67% with 518 orphan rows and 14.67% with 2,072.
- **Gold density** — FiQA has 2.6 gold pages per query, 11.5× fewer than scidocs' 30, yet its
  PER (5.34%) is *lower* than scidocs' (7.12%).

The manuscript states this as an unexplained quantity rather than a hypothesis.

## Environment

The scripts use only the Python standard library, except `check_numbers.py` and `verify.py`,
which import `pypdf` for PDF text extraction. Tested on Python 3.12.

Experiments were run on a 131,028-chunk Chinese technical knowledge base, with external
replication on BEIR and C-MTEB subsets and a cross-platform re-run on an NVIDIA RTX 5060.
Compute for the main experiments was provided by the National Supercomputing Center in
Zhengzhou (NSCC-ZZ).

## License

Data and scripts: MIT. Figures: CC BY 4.0.

## Timestamp (RFC 3161)

`MANIFEST-SHA256.txt` records the digest of every file in this archive. To fix *when* that manifest
existed, independent of our own clocks, it was submitted to a public Time-Stamp Authority:

| | |
|---|---|
| Query | `MANIFEST-SHA256.tsq` |
| Reply | `MANIFEST-SHA256.tsr` |
| TSA | freetsa.org |
| Granted | **2026-11-18 09:34:26 GMT** |
| Serial | `0x0A25EBE0` |

A TSA signs the digest together with a trusted time; it does not endorse the content. What this
establishes is that the manifest above existed in this form no later than that instant. It does not
establish what happened before.

To verify, fetch the freetsa CA and TSA certificates, then:

```bash
openssl ts -verify -in MANIFEST-SHA256.tsr -queryfile MANIFEST-SHA256.tsq \
  -CAfile cacert.pem -untrusted tsa.crt
```

Expected: `Verification: OK`.
