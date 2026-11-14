# E1 预注册协议（跑前冻结，不得事后修改）

任务：`t2 — E1 偏斜量化：五类更新事件 × SVR/PER/RDU`（团队 scheda-paper）
冻结时间：2026-11-10 19:35 (+08:00)，**在第一次评测运行之前**写下；运行脚本会把本文件的
sha256 写进 `run-meta.json`，`notes.md` 引用同一摘要。任何改动都必须新建 `PROTOCOL-v2.md`
并重跑，不得改本文件后复用旧结果。

---

## 1 预注册判据（原文照抄任务书，逐字冻结）

> 判据跑前写死、不得事后改：重分块使 PER 上升 ≥15pp 记 severe；其它事件 <3pp 如实写"影响有限"。

操作化（三处细节在此写死，避免事后解释）：

- **C1（严重性）**：`rechunk` 事件的**满幅档（m=1.00，全库重分块）**使 `PER@5` 相对基线上升
  **≥ 15.0 个百分点** ⇒ 记 `severe`；< 15.0pp ⇒ 记 `not_severe`。10% 档只作幅度响应，不参与 C1 判定。
- **C2（其余四类）**：`add / delete / modify / model_change` 的**每一档**若 `PER@5` 上升
  **< 3.0 个百分点** ⇒ 该档写「影响有限」；若 ≥ 3.0pp ⇒ **如实报告该档数值并标注 C2 未满足**，
  不得回头调高阈值、不得只报满幅档。
- **C3（不因结果好坏改口径）**：指标定义、事件构造、抽样种子、基线口径在本文件中冻结；
  若实现与本文不符，以本文为准修实现，不以实现回改本文。

## 2 数据与基线口径

| 项 | 值 / 来源 |
|---|---|
| 索引 | `$KB_ROOT/index/kb.sqlite`（`meta` 表，逐块 `chunk_id, path, startline, text`） |
| 分块现状 | `lines` 策略：固定 15 行不重叠窗口（`startline` 步长 15）；粗块 = 0 行补齐（尾块可短） |
| 块身份 | `chunk_id = sha256(path + "\x1f" + text)[:16]`（与 `scheda/index.py:chunk_id_of` 逐字节一致，已复现） |
| 语料量 | 79,867 块 / 11,159 页（镜像 `$KB_ROOT/mirror`，15,185 文件，索引内 11,159 页） |
| 向量库 | `$KB_ROOT/index/vectors.sqlite`，`model=bge-m3`，1024 维，79,867 行，**当前 0 孤儿** |
| 第二世代 | `$KB_ROOT/index/vectors-qwen3.sqlite`，`qwen3-embedding-0.6b-int8`，1024 维，14,055 行（其中 9,347 行的 `chunk_id` 仍在当前索引里） |
| 评测集 | `$KB_ROOT/bench/gold.sqlite` → `experiments/zzu-pilot/gold.json`：192 题，**109 题可计分**（`usable=1` 且有 `gold_src` span） |
| 基线评测脚本 | `experiments/zzu-pilot/eval.py`（BM25 + 稠密 + RRF(k=60) + 页级聚合 + 0.002·min(命中数,5) 页内加成） |
| 基线数值（必须复现） | `recall@5 = 0.27522935779816515`，`mrr@10 = 0.17671472258628215`（109 题） |

**基线复核闸门**：本次实现的 `baseline` 状态必须逐位复现上述两个数，否则实验不成立，
先修实现再跑（闸门写在 `run.py` 里，不通过直接非零退出）。

## 3 被测量的系统："朴素 upsert"（改造前的旧行为）

E1 量的是**修复前的系统**，不是改造后的协议。旧行为按 `ragvec.py` 历史实现与
`20261106` 结项报告的描述复刻：

1. 向量库**全表加载**，从不校验 `chunk_id` 是否仍在当前索引里（报告的原始缺陷）。
2. 每条向量行自带 `(path, startline)`，命中后**按行自己的坐标归页** —— 于是孤儿行仍然
   投票给它所属的页面。这正是"96% 孤儿仍在参与排序融合"的机制。
3. 事件发生后的状态是**修复前的状态（pre-repair）**：刷新窗口内不重嵌、不清理。
   本窗口内禁止新增嵌入（`decision-embedding.md`：131k 块 ≈ 9.5 h，且生产每小时 :37 要补嵌），
   因此"事件后立刻重嵌"的状态**不在 E1 范围内**（那是 E2/E3 的修复侧）。

搜索路径与 `eval.py` 完全一致：BM25 建在当前索引块上（词法臂永远是最新文本），
稠密臂建在**整个向量库**上（含孤儿），两臂 RRF 融合到页级，取 top-10 计指标。

## 4 指标定义（全部在此写死）

记一次评测的最终页级 top-10 为 `L = [p1..p10]`，题目集合 `Q`（默认 109）。

### 4.1 SVR（存储层，定义 2 的操作化）

```
SVR = |{ row ∈ V : row.chunk_id ∉ ids(C) }| / |V|
```
`C` = 当前索引块集合，`V` = 向量库全部行。**确定性量，无抽样误差，不给 CI**（在表里标 `n/a`）。

### 4.2 SVR_stale（定义 1(b) 的操作化，SVR 的补充量）

```
SVR_stale = |{ row ∈ V : chunk_id 活, 但 (row.model ≠ 当前查询模型) }| / |V|
```
只有"换模型/内容改写"才会让它非零 —— 这类偏斜**在 SVR 里看不见**，是本实验的核心对照。

### 4.3 PER@5（来源错误率，定义 3 的操作化；**主指标**）

对每个问题 q、每个进入 top-5 的页 p，判定 p 的**支撑证据**是否错位：

- 稠密臂对页 p 的支撑证据 = 稠密 top-50 里**排名最高**的那条属于 p 的行（记为 `r*(p)`）。
  （页级 RRF 对同页多块取 max，因此这条行就是真正把 p 顶上去的那条证据。）
- 行 `r` 判为 **tainted** 当且仅当：
  - (a) `r.chunk_id ∉ ids(C)` —— 孤儿行（陈旧块，定义 1(a)）；或
  - (b) `r.chunk_id ∈ ids(C)` 但 `r.model ≠ 当前查询模型` —— 陈旧世代（定义 1(b)）。
- 若 `r*(p)` tainted ⇒ 页 p tainted；若 p 没有任何稠密支撑（纯词法命中） ⇒ p 不 tainted
  （词法臂永远读当前文本）。
- 死页（`path ∉ 当前索引页集合`）必然只能由孤儿行支撑 ⇒ tainted。

```
PER@5 = (# tainted 的 (q, p) 对, p ∈ top-5) / (5 · |Q|)
```
基线 PER@5 = **0**（当前库 0 孤儿、模型一致），这正是定义 3 想要的"恰为 0"语义。

### 4.4 PER_gold@5（次要口径，防定义 3 被字面误读）

`PER_gold@5 = 1 − (top-5 中落在 gold span 上的页数)/5`。只作为**诊断列**写进 CSV，
不参与 C1/C2 判定（否则基线就 ≈0.9，判据失去意义；此处**事先声明**这一口径分歧）。

### 4.5 RDU（定义 4）

```
RDU = (R0 − R1) / R0,   R = recall@5（同一题目集合）
```
同时报告 `Δrecall@5` 的百分点值与 95% CI。

### 4.6 受影响/未受影响拆分（所有事件都算，事前声明）

- `affected(Q)` = gold span 的任一 `path` 被该事件触碰过的问题；`unaffected(Q)` = 其余。
- 主表 PER/RDU 用全量 Q；另给 `rdu_unaffected`、`per5_unaffected` 用于把"信息损失"与
  "偏斜的连带伤害"分开（删除/改写类事件必须这样拆，否则 RDU 会被信息损失主导）。

## 5 事件定义（五类 × 两档，全部确定性、seed=20261110）

**事件目标集 T 与幅度 m 的口径（事前写死）**：结构型事件（重分块、换模型）的每页代价无界
（会改掉该页全部块），因此其目标集就是全库；文档型事件（新增、删除、修改）以**一批文档**
为单位，目标集 T = 全库页面的 10%（随机抽样）。幅度 m ∈ {0.10, 1.00} = **T 中被真正触碰的比例**。
表里同时给 `affected_pages / affected_chunks / affected_chunk_frac`，读者按绝对规模比较，
不按名义幅度比较。

**候选页**：镜像里按 `lines(15)` 重切后与索引逐块一致（`chunk_id` 集合完全相同）的页面 ——
共 N_elig 页（`export_base` 里算，非候选页不参与任何事件，见笔记里的计数）。

1. **`add`（新增）**：T = 从"与评测无关"的候选页抽 10%（排除任何 gold span 页、排除基线 top-10 出现过的页）。
   触碰页的正文**原样复制**到新路径 `zz-e1-add/<i:04d>/<原路径>` 并重新切块（块身份含 path ⇒ 全部是新块，
   库里没有它们的向量：窗口内不重嵌的写实状态）。`C ← C ∪ 新块`，`V` 不变。
   *（补充行 `add_real`：把镜像里 56 个未入索引的 `z1-restricted/*.md` 真实文档加进索引，同一机制。）*

2. **`delete`（删除）**：T = 候选页随机抽 10%（**包含** gold 页，便于拆 affected/unaffected）。
   触碰页的块从 `C` 移除；`V` 不变（旧 upsert 从不删除 ⇒ 孤儿）。
   幅度 m=1.00 时删掉 T 的全部（≈10% 页面）。

3. **`modify`（改写）**：T = 候选页随机抽 10%。触碰页做**模板迁移**：每一行加 `"> "` 前缀
   （Markdown 引用化）。行数不变 ⇒ gold span 仍有效；`TOK/CJK` 切词不变 ⇒ **词法臂逐位相同**，
   只有块内容哈希变了。`C ← 触碰页用 lines(15) 重切的新块`；`V` 不变（旧行成孤儿）。
   *（补充行 `mod_append`：触碰页末尾追加一节真实更新文字，模拟"只动页尾"的常见编辑。）*

4. **`rechunk`（重分块）**：T = 全部候选页。触碰页改用 `structural` 策略重切
   （`chunk_strategy=structural, chunk_max_chars=1000`，即 `scheda/index.py:_chunk_structural` 原实现；
   与 20261106 那次把索引从 75,821 撑到 131,028 块的历史事件同源）。`V` 不变 ⇒ 旧块全成孤儿。
   m=0.10 = 抽 10% 页面重分块；m=1.00 = 全库重分块。
   *（补充行 `rechunk_lines20`：全库把窗口 15→20 行，参数型重分块，供 §6.4 敏感性用。）*

5. **`model_change`（换模型）**：**面板受限，事前声明**。第二世代 qwen3 向量只覆盖 9,347 块
   （占语料 11.7%，涉 826 页、83 道可计分题）。面板 = 这 826 页的全部 10,227 块建索引 +
   这 9,347 块作向量库（`model` 字段区分世代）；`Q_panel = 83` 题。
   - `panel_base`：全部行 bge-m3，查询 bge-m3 ⇒ PER=0。
   - m=0.10 / 1.00：把面板向量库里 **10% / 100% 的行**换成 qwen3 向量（`model` 字段=新世代），
     查询侧同时切到 qwen3（即"查询侧先切、存储未迁完"的朴素顺序）。`SVR` 恒为 0。
   - 全库口径的 `SVR=0`、`SVR_stale`（= 已迁移比例）照常报告；`PER/RDU` 只在面板内报告（n=83）。
   - 已知局限（事前声明）：查询侧 qwen3 编码器与存量 qwen3 向量的**实现漂移**（同一文本 cos≈0.88），
     只影响 recall 绝对值、不影响 PER 的世代记账口径；`m=1.00` 的一致性结论（PER=0）不受影响。

## 6 统计（事前写死）

- **配对 bootstrap**：对题目下标做 1000 次有放回重采样（`seed=20261110`），**同一重采样**同时用于
  基线与事件态（配对），取 2.5/97.5 分位给 95% CI。`PER@5`、`ΔPER@5`、`recall@5`、`Δrecall@5`、
  `RDU` 都给 CI。SVR 是确定量，CI = n/a。
- **Spearman ρ（主分析）**：在 10 个主行（5 类 × 2 档）上，`SVR` 与 `PER@5` 的 Spearman ρ；
  附 1000 次置换检验的 p 值。
- **Spearman ρ（次要，事前声明）**：同样 10 行但剔除 `model_change` 两行后的 ρ；
  以及 `SVR_stale` 与 `PER@5` 的 ρ。（模型事件正是"破坏相关性"的那一类，必须单独给出。）
- 不做 Holm 校正（E1 只有 1 个确认性判据 C1、4 个描述性判据 C2，不构成多重比较族）。

## 7 产出物

- `experiments/e1/table1.csv` —— 每行一个状态：`event, magnitude, kind(primary|supplementary|baseline),
  scope, n_questions, target_pages, affected_pages, affected_chunks, affected_chunk_frac,
  svr, svr_stale, per5, per5_lo, per5_hi, dper5_pp, dper5_pp_lo, dper5_pp_hi,
  per_gold5, recall5, recall5_lo, recall5_hi, drecall5_pp, drecall5_pp_lo, drecall5_pp_hi,
  rdu, rdu_lo, rdu_hi, mrr10, rdu_unaffected, per5_unaffected, notes`
- `experiments/e1/fig2.csv` —— 图 2 散点数据（每点一个状态）：`label, event, magnitude, kind, svr,
  svr_stale, per5, per5_lo, per5_hi, n_questions`
- `experiments/e1/raw/per_query.csv.gz` —— 每状态 × 每题的原始判定（recall5、per 槽位、top5 页、
  tainted 页、affected 标记），供复核与重算
- `experiments/e1/notes.md` —— 结果描述（可直接进论文 §6.1）+ 判据满足情况 + 局限
- `experiments/e1/protocol.sha256`、`run-meta.json`、`stats.json`

## 8 环境

本机 SER8（16 核 Zen4，无独显），Python 3.12，numpy；`PYTHONPATH=$KB_LIB`
（numpy）；分块器直接 import `scheda.index._chunk_page`（release/src，与生产同源）。
全部只读打开生产库（`mode=ro`），不写 `$KB_ROOT/**`。
