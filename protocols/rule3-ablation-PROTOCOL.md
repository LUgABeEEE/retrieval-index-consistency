# 规则 3 消融 · 预注册（跑前冻结）

任务 t19。冻结时间 2026-11-11 17:25 (+08:00)，写在**任何评测之前**。

## 1 被测机制

生产 `release/src/scheda/ragvec.py: VectorArm.load()` 的缓存键：
- **规则 3（正确）**：`(向量库路径, 向量库 mtime, 向量库 size, 索引身份三元组 mtime_ns/size/inode)`。
- **改造前（错误）**：只含向量库 mtime（与 size）。重分块只改 `kb.sqlite`、不改 `vectors.sqlite`，
  所以旧矩阵会被原样复用。

## 2 两个状态

| 状态 | 索引 | 向量库 |
|---|---|---|
| A | 生产 15 行切块，79,867 块（`experiments/e1/cache/base.pkl`） | A 的 bge-m3 向量，79,867×1024（`e1/cache/vecs_bge.npy`） |
| B | structural 切块，131,028 块（`e2/corpus-structural.jsonl.gz`） | B 的 bge-m3 向量，131,028×1024（`e2/vectors/vectors.npy`） |

## 3 四组（+2 个补充组）

1. **A 基准**：索引 A + 向量库 A（正确配对），冷缓存。
2. **B 基准**：索引 B + 向量库 B，冷缓存。
3. **C1（错误键）**：向量库 B；`store_mtime` 恒定（模拟"重分块不写向量库"）；
   r0 = 索引 A（建矩阵并缓存），r1–r3 = 索引 B（**缓存命中，沿用 r0 的矩阵**）。
4. **C2（正确键）**：同上，但键含索引身份 ⇒ r1 失效重建。
5. **补充 C1-noreembed（生产真实路径）**：向量库 A；r0 = 索引 A（缓存 A 的完整矩阵），
   r1–r3 = 索引 B **不重嵌**（缓存命中，旧矩阵 79,867 行仍在，其中绝大多数对 B 是孤儿）。
6. **补充 crash-probe（回答任务第 5 点）**：向量库 B；r0 = 索引 B（缓存 131,028 行），
   再切到索引 A（79,867 块）——看"旧矩阵行数 > 新索引块数"时，按行下标耦合会不会越界报错。

## 4 指标

`recall@5` / `MRR@10` / `PER@5`（METRICS v2 口径：页面级 + 真实跨度；PER = 证据行坐标/文本错位比例）、
`n_rows_loaded`（本次加载后的矩阵行数）、`build_ms`（load 耗时）、`cache_hit`。
"首轮 + 前 3 轮" = r1/r2/r3；无状态变化时 r2/r3 与 r1 逐位相同，仍调用 load 记录命中与耗时。

## 5 判据（写死，不得调阈值）

**若 |C1 − C2| 的 recall@5 差 ≥ 1pp 或 PER@5 差 ≥ 1pp ⇒ 记「规则 3 有效」；否则记「防御性」。**

## 6 产出

`experiments/rule3-ablation/table.csv`（每行 = 组 × 轮次）与 `notes.md`（含判据结论、崩溃 vs 静默的实测、局限）。
