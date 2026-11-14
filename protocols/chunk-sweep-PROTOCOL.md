# 切块大小扫描 · 预注册协议（跑前冻结）

任务：`t11 — 切块大小扫描：10/15/20/30/40 行 × 质量/失效/成本`（团队 scheda-paper，执行者 exp-e1）
冻结时间：2026-11-11 11:30 (+08:00)，**在任何切块/嵌入/评测之前写下**。运行脚本把本文件 sha256
写进 `run-meta.json`，`notes.md` 引用同一摘要。改设计必须新建 `PROTOCOL-v2.md` 并重跑。

---

## 1 要回答的三个问题

1. **检索质量**：块大小如何影响 recall@5 / span@5 / MRR@10 / PER？
2. **失效幅度**：同一事件（全库改切为 structural）下，哪一档的 SVR / PER / RDU 更小？
3. **修复成本**：每档的块数、嵌入耗时、p95 延迟。

## 2 语料与切块（固定页面集合）

- **页面集合**：当前生产索引（`lines` 15 行，79,867 块）覆盖的 **11,159 页**，页面原文取自
  `$KB_ROOT/mirror`。这就是"固定 79,867 块那版语料的页集合"，跨档不变。
- **每档切块**：同一份 `scheda/index.py:_chunk_page`，`chunk_strategy=lines`，`chunk_lines ∈ {10,15,20,30,40}`。
  块身份 `chunk_id = sha256(path + 0x1f + text)[:16]`。
- **一致性自检**：15 行档重切后与生产索引逐块比对；已知 166 个块（0.21%）因 mirror 快照晚于索引构建
  而不可复现，**只记录、不修补**（不拿生产索引直接当 15 行档，避免跨档语料来源不一致）。

## 3 向量（5 卡时一致的唯一配置）

- 模型：bge-m3 **fp16**，4 卡 DCU（`experiments/hpc/embed_dcu_mp.py` 同源实现，按 rank 交错分片后
  **按 item 下标重排**再存储；重排映射同时落盘供审计）。
- **maxlen = 512**（与 METRICS.md v2 / 生产一致）。因此 30 行以上会有块被截断 —— 这**是生产行为的一部分**，
  每档的截断率必须报出；另加一档 `40@1024` 作"截断 vs 块太大"的对照（补充行，不参与 C1–C4）。
- 向量 L2 归一，存 float16（与模型输出精度一致），评测时转 float32。
- 查询向量 `experiments/zzu-pilot/qvecs.npy`，**按 qid 取**（METRICS.md §2 的硬规定）。

## 4 指标（METRICS.md v2，一字不改）

| 指标 | 定义 |
|---|---|
| recall@5 | top-5 页中存在 `p==gold.path` 且该页有块跨度与 gold 窗口相交 |
| span@5 | top-5 页里，**返回的证据块/证据行**的真实跨度与 gold 窗口相交 |
| MRR@10 | 第一个相关页的倒数排名 |
| PER@5 / PER@10 | 证据行 `(path,startline)` 不在存活块集、或该坐标文本 ≠ 行内文本 的比例 |
| p50/p95 | 每 query 墙钟（两臂 + 融合 + 页聚合），单进程，本机 16 核 |
| 跨度 | `[startline, startline+块内换行数]`，1-based 闭区间 |

每档的基线状态是"库与索引一致"（SVR = 0、PER = 0）。

## 5 重分块事件（失效幅度）

对每一档 s：事件后索引 = 同一页集合改用 `structural`（`chunk_max_chars=1000`）切块；
向量库**保持该档旧向量不重嵌**（朴素 upsert + 修复前状态，与 E1 同一语义）。
- `SVR_after_rechunk` = 旧行中 chunk_id 不在新索引里的比例；
- `rdu` = (recall0 − recall1)/recall0；同时报 PER@5 / span@5 / p95。

## 6 预注册判据（写死，不因结果改）

- **C1（小块优先）**：若 10 行档同时满足 `recall@5 ≥ 15 行档 − 1.0pp` **且** `PER@5 ≤ 15 行档 + 1.0pp`，
  记"小块不劣、小块优先"成立；否则记不成立。
- **C2（块越大越差）**：若 recall@5 随 size 单调不增（Spearman ρ(size, recall) ≤ −0.8 且置换 p < 0.05），
  记"经典权衡成立"；否则记"非单调"，并指出峰值档。
- **C3（小块失效面积更大）**：若 `SVR_after_rechunk` 随 size 单调不增（ρ(size, SVR) ≤ −0.8），
  记"小块在重分块事件下失效面积更大"成立；否则如实写。
- **C4（成本近似线性于块数）**：若 5 档 `embed_secs / n_chunks` 的极差 ≤ 20%，记"固定 maxlen 下成本线性于块数"；
  否则如实写（短块 padding 会放大单位成本）。
- **反例条款**：任何"更小块反而更差"（10 行档 `recall@5 < 15 行档 − 1.0pp` 或 `PER@5 > 15 行档 + 1.0pp`）
  必须单列指出，不得被"整体趋势"掩盖。
- **C1–C4 只用 5 个主档（10/15/20/30/40）判定**；`40@1024` 只作解释性对照。

## 7 统计

配对 bootstrap 1000 次（seed=20261110）给各档 recall@5 / span@5 / PER@5 的 95% CI 与相对 15 行档的 ΔCI；
Spearman ρ 用置换检验（2000 次）。n = 109 题。

## 8 局限（事前声明）

1. maxlen=512：30/40 行档有块被截断，质量下降里混着"截断"效应；`40@1024` 对照只消解 40 行档这一处。
2. 向量由 DCU fp16 生成，与产线 Ollama F16 非同一实现 ⇒ **15 行档的绝对值不能直接与 METRICS 卡的
   0.486239 并列**；跨档比较在同一实现内有效。会给出 15 行档 DCU vs Ollama 的余弦抽检。
3. 事件只测一种（structural 重切）；换窗口大小的事件见 E1 的 `supp_rechunk_lines20`。
4. 修复成本只报"全量重建该档"的嵌入耗时，不含调度（E3）。

## 9 产出

`experiments/chunk-sweep/table.csv`（列：size_lines, n_chunks, avg_len, avg_tokens, trunc_rate, recall5,
span5, mrr10, per5, per10, svr_after_rechunk, rdu, embed_secs, p50_ms, p95_ms, ci 列…）、`fig.svg`、
`notes.md`、`vectors/size{10,15,20,30,40}/vecs.npy` + `meta.json`、`corpora/*.jsonl.gz`、`run-meta.json`。
