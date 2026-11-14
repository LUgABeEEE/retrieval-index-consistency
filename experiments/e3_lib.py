#!/usr/bin/env python3
"""E3 公共库：预算受限修复调度的数据装载、检索复刻、六种策略与配对统计。

口径唯一来源：`experiments/METRICS.md` v2（0.486239 基线，页面级跨度重叠 rec@5）。
本文件不实现评测流程本身（那是 run_e3.py），只提供可复用构件：
  * 语料 / 向量 / 查询装载（qvec **按 qid 对齐**，见 METRICS §2）
  * BM25 top-50 缓存 + 稠密全量分数矩阵 S = Q·Vᵀ 缓存（一次算好，反复掩码评测）
  * FastEval：给定「已嵌入块集合」算 recall@5，并给出历史访问事件的 top-10 页
  * 策略排序：FIFO / LRU / FREQ / RANDOM / GREEDY（子模贪心，见 PROTOCOL.md §4）
  * exact Wilcoxon signed-rank（n 小，不用 scipy）

设计要点（与 E2 的关系）
  * 候选池 = 当前存活块集 ∩ 已嵌入集。E3 的块身份就是语料行号，不存在 naive 陈旧行；
    E2 的 `ProtocolStore.matrix()/meta()` 是同样的「按存活集过滤」语义。
  * 词法器一次建好、所有策略共用（重分块后词法索引立即重建，生产就是这样）。
  * 评测时只把「未嵌入块」的稠密分数掩成 -inf，其余完全按 METRICS §3 的 RRF + 页聚合。
"""
from __future__ import annotations

import collections
import gc
import gzip
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / 'e2'))          # 复用 E2 的分词/BM25/语料装载
sys.path.insert(0, str(HERE.parent / 'hpc' / 'pylibs'))

import numpy as np  # noqa: E402

import e2_lib as L  # noqa: E402

RRF_K = 60
DEPTH = 50
PAGE_BONUS = 0.002


# --------------------------------------------------------------------------- #
# 语料
# --------------------------------------------------------------------------- #

class Corpus:
    """块表 + 页表。行序就是向量行序（E2 的 vectors.npy 与语料逐行对应）。"""

    def __init__(self, ids, paths, starts, texts):
        self.ids = ids
        self.paths = paths
        self.starts = np.asarray(starts, dtype=np.int32)
        self.texts = texts          # 只在建 BM25 / 跨度时用，建完可 drop_texts()
        self.n = len(ids)
        self.page_names = sorted(set(paths))
        pid = {p: i for i, p in enumerate(self.page_names)}
        self.page_ids = np.asarray([pid[p] for p in paths], dtype=np.int32)
        self.n_pages = len(self.page_names)
        self.page_chunks: list[list[int]] = [[] for _ in range(self.n_pages)]
        for i, p in enumerate(self.page_ids):
            self.page_chunks[int(p)].append(i)

    def drop_texts(self):
        self.texts = None
        gc.collect()


def load_corpus(path: str) -> Corpus:
    ids, paths, starts, texts = [], [], [], []
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            ids.append(rec['id'])
            paths.append(rec['path'])
            starts.append(int(rec['start']))
            texts.append(rec['text'])
    return Corpus(ids, paths, starts, texts)


def load_corpus_from_pickle(path: str) -> Corpus:
    """E1 旧缓存 `base.pkl`（chunks=[(cid,path,startline,text)]，真实换行）——仅干跑用。"""
    import pickle
    base = pickle.load(open(path, 'rb'))
    rows = base['chunks']
    return Corpus([r[0] for r in rows], [r[1] for r in rows],
                  [r[2] for r in rows], [r[3] for r in rows])


def load_gold(path: str):
    gold = json.load(open(path, encoding='utf-8'))
    qs = [q for q in gold['questions'] if q['qid'] in gold['qrels']]
    if len(qs) != 109:
        raise SystemExit(f'可计分题数 {len(qs)} != 109（METRICS §2）')
    return gold, qs


def align_qvecs(qvec_path: str, gold: dict, qs: list[dict], qids: list[str] | None = None) -> np.ndarray:
    """按 qid 对齐查询向量（METRICS v2 §2 的唯一正确入口；等价 E2 的 align_qvecs()）。"""
    qv = np.load(qvec_path).astype(np.float32)
    pos = {q['qid']: i for i, q in enumerate(gold['questions'])}
    use = qs if qids is None else [q for q in qs if q['qid'] in set(qids)]
    return np.stack([qv[pos[q['qid']]] for q in use])


def page_relevant_pages(c: Corpus, gold: dict, qs: list[dict]) -> list[set[int]]:
    """每题的相关页集合：路径命中 且 该页有块跨度与任一 gold 窗口重叠（METRICS §4）。

    页相关性只取决于「存活块集」，与哪些块已嵌入无关 ⇒ 一次算好，评测时 O(1) 查。
    """
    pid = {p: i for i, p in enumerate(c.page_names)}
    spans_by_page: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for i in range(c.n):
        t = c.texts[i]
        spans_by_page[int(c.page_ids[i])].append((int(c.starts[i]), int(c.starts[i]) + t.count('\n')))
    out = []
    for q in qs:
        rel: set[int] = set()
        for sp in gold['qrels'].get(q['qid'], []):
            p = pid.get(sp['path'])
            if p is None:
                continue
            for lo, hi in spans_by_page.get(p, ()):
                if lo <= sp['hi'] and hi >= sp['lo']:
                    rel.add(p)
                    break
        out.append(rel)
    return out


# --------------------------------------------------------------------------- #
# BM25 top-50 与稠密分数矩阵（缓存）
# --------------------------------------------------------------------------- #

def file_sig(path: str) -> str:
    """文件签名：大小 + 头/尾各 1 MiB 的 sha256（537 MB 的 vectors.npy 也秒级）。"""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read(1 << 20))
        if size > (1 << 21):
            f.seek(size - (1 << 20))
            h.update(f.read(1 << 20))
    return f'{size}-{h.hexdigest()[:16]}' 


def bm25_topk(bm25: 'L.BM25', questions: list[str], depth: int = DEPTH) -> np.ndarray:
    out = np.full((len(questions), depth), -1, dtype=np.int32)
    for i, q in enumerate(questions):
        hits = bm25.search(q['question'], depth)
        for j, (d, _s) in enumerate(hits):
            out[i, j] = d
    return out


def build_bm25_topk(c: Corpus, questions: list[str], cache: Path, tag: str,
                    depth: int = DEPTH) -> np.ndarray:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        z = np.load(cache)
        if z['idx'].shape == (len(questions), depth) and str(z['tag']) == tag:
            print(f'[bm25] cache hit {cache.name} ({z["seconds"]:.0f}s to build)', flush=True)
            return z['idx']
    t0 = time.time()
    bm25 = L.BM25(c.texts)
    idx = bm25_topk(bm25, questions, depth)
    sec = time.time() - t0
    np.savez_compressed(cache, idx=idx, tag=tag, seconds=sec)
    print(f'[bm25] built in {sec:.1f}s → {cache.name}', flush=True)
    del bm25
    gc.collect()
    return idx


def build_dense_scores(qv_all: np.ndarray, V, cache: Path, tag: str,
                       block: int = 8192) -> np.ndarray:
    """S = qv_all · Vᵀ，float32，(Q, N)。Q 全部 136 行一次算完，评测时按行取。"""
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        S = np.load(cache, mmap_mode='r')
        if S.shape == (qv_all.shape[0], V.shape[0]):
            print(f'[dense] cache hit {cache.name} {S.shape}', flush=True)
            return S
    t0 = time.time()
    N = V.shape[0]
    S = np.empty((qv_all.shape[0], N), dtype=np.float32)
    for lo in range(0, N, block):
        hi = min(lo + block, N)
        S[:, lo:hi] = qv_all @ np.asarray(V[lo:hi], dtype=np.float32).T
    np.save(cache, S)
    print(f'[dense] S={S.shape} in {time.time()-t0:.1f}s → {cache.name}', flush=True)
    return S


# --------------------------------------------------------------------------- #
# 评测
# --------------------------------------------------------------------------- #

class FastEval:
    """给定「已嵌入块下标数组」评测页面级 recall@5；也用于生成历史访问事件。"""

    def __init__(self, S, bm25_idx: np.ndarray, c: Corpus, rel_pages: list[set[int]]):
        self.S = S
        self.bm = bm25_idx          # (Q, depth) 语料行号，-1 填充
        self.c = c
        self.rel = rel_pages        # 与 bm25_idx 行同序
        self.nq = bm25_idx.shape[0]
        self.page_of = c.page_ids

    def _rank(self, qi: int, embedded: np.ndarray):
        kw = self.bm[qi]
        dn: list[int] = []
        if embedded.size:
            sims = self.S[qi, embedded]
            k = min(DEPTH, sims.shape[0])
            if k == sims.shape[0]:
                order = np.argsort(-sims)
            else:
                order = np.argpartition(-sims, k - 1)[:k]
                order = order[np.argsort(-sims[order])]
            dn = embedded[order].tolist()
        fused: dict[int, float] = {}
        for r, d in enumerate(kw.tolist()):
            if d < 0:
                break
            fused[d] = fused.get(d, 0.0) + 1.0 / (RRF_K + r + 1)
        for r, d in enumerate(dn):
            fused[d] = fused.get(d, 0.0) + 1.0 / (RRF_K + r + 1)
        pages: dict[int, float] = {}
        cnt: dict[int, int] = collections.defaultdict(int)
        for d, s in fused.items():
            p = int(self.page_of[d])
            if s > pages.get(p, 0.0):
                pages[p] = s
            cnt[p] += 1
        for p in pages:
            pages[p] += PAGE_BONUS * min(cnt[p], 5)
        return sorted(pages.items(), key=lambda kv: -kv[1])[:10]

    def recall(self, embedded: np.ndarray, qrows: list[int]) -> np.ndarray:
        """返回每题 0/1 的 recall@5（页面级跨度重叠，METRICS §4）。"""
        out = np.zeros(len(qrows), dtype=np.int8)
        for j, qi in enumerate(qrows):
            order = self._rank(qi, embedded)
            relq = self.rel[qi]
            out[j] = 1 if any(p in relq for p, _ in order[:5]) else 0
        return out

    def top_pages(self, embedded: np.ndarray, qi: int) -> list[int]:
        return [p for p, _s in self._rank(qi, embedded)]

    def top_chunks_of_pages(self, embedded: np.ndarray, qi: int, pages: set[int]) -> list[int]:
        """用于历史证据计数：返回该 query 结果中落在 pages 里的命中块（词法/稠密）。"""
        order = self._rank(qi, embedded)
        keep = {p for p, _ in order if p in pages}
        hits = []
        for d in self.bm[qi].tolist():
            if d < 0:
                break
            if int(self.page_of[d]) in keep:
                hits.append(d)
        if embedded.size:
            sims = self.S[qi, embedded]
            k = min(DEPTH, sims.shape[0])
            part = np.argpartition(-sims, k - 1)[:k] if k < sims.shape[0] else np.arange(k)
            part = part[np.argsort(-sims[part])]
            for j in part.tolist():
                d = int(embedded[j])
                if int(self.page_of[d]) in keep:
                    hits.append(d)
        return hits


# --------------------------------------------------------------------------- #
# 策略
# --------------------------------------------------------------------------- #

def order_fifo(c: Corpus) -> np.ndarray:
    return np.arange(c.n, dtype=np.int64)


def order_random(c: Corpus, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(c.n).astype(np.int64)


def page_rank_from_events(c: Corpus, events: list[tuple[int, list[int]]],
                          mode: str) -> list[int]:
    """events = [(timestamp, [page_id top10])…] → 页的优先级序列。

    LRU: 最近被访问的页优先（按最后时间戳降序）；
    FREQ: 访问次数降序，同频按最后时间戳降序；两种都按 page_id 升序兜底。
    """
    cnt: collections.Counter = collections.Counter()
    last: dict[int, int] = {}
    for ts, pages in events:
        for p in pages:
            cnt[p] += 1
            last[p] = max(last.get(p, -1), ts)
    if mode == 'lru':
        key = lambda p: (-last.get(p, -1), p)
    elif mode == 'freq':
        key = lambda p: (-cnt.get(p, 0), -last.get(p, -1), p)
    else:
        raise ValueError(mode)
    return sorted(range(c.n_pages), key=key)


def order_by_pages(c: Corpus, page_order: list[int],
                   evidence: np.ndarray | None = None) -> np.ndarray:
    out: list[int] = []
    for p in page_order:
        ch = c.page_chunks[p]
        if evidence is not None and len(ch) > 1:
            ch = sorted(ch, key=lambda i: (-int(evidence[i]), int(c.starts[i])))
        out.extend(ch)
    return np.asarray(out, dtype=np.int64)


def order_greedy(c: Corpus, W: np.ndarray, evidence: np.ndarray | None = None,
                 chunk_key: np.ndarray | None = None) -> np.ndarray:
    """子模贪心：U(S)=Σ_p W(p)·sqrt(n_S(p)/N_p)（W 由调用方定义，可为标注或纯日志）。

    边际增益 Δ(p) = W(p)·[sqrt((n+1)/N_p) − sqrt(n/N_p)]，单调递减。
    用堆做惰性贪心：每次取当前边际最大的页、补它的下一块、更新后重新入堆。
    页内块序：evidence 非空 → 历史证据次数降序；否则 chunk_key 非空 → 该分数降序；
    最后按行号兜底。`W` 换成无标注权重即成 no-label 变体（B-M1）。
    """
    import heapq
    N_p = np.array([len(ch) for ch in c.page_chunks], dtype=np.float64)
    heap: list[tuple[float, int]] = []
    for p in range(c.n_pages):
        w, n = W[p], N_p[p]
        if w > 0 and n > 0:
            gain = w * (math.sqrt(1.0 / n))
            heapq.heappush(heap, (-gain, p))
    used = np.zeros(c.n_pages, dtype=np.int32)
    out: list[int] = []
    chunk_seq: list[list[int]] = [None] * c.n_pages  # type: ignore
    while heap:
        neg, p = heapq.heappop(heap)
        ch = chunk_seq[p]
        if ch is None:
            ch = c.page_chunks[p]
            if evidence is not None and len(ch) > 1:
                ch = sorted(ch, key=lambda i: (-int(evidence[i]), int(c.starts[i])))
            elif chunk_key is not None and len(ch) > 1:
                ch = sorted(ch, key=lambda i: (-float(chunk_key[i]), int(c.starts[i])))
            chunk_seq[p] = ch
        n = used[p]
        out.append(ch[n])
        used[p] = n + 1
        w, N = W[p], N_p[p]
        if used[p] < N:
            m = used[p]
            gain = w * (math.sqrt((m + 1) / N) - math.sqrt(m / N))
            heapq.heappush(heap, (-gain, p))
    mask = np.zeros(c.n, dtype=bool)
    mask[np.asarray(out, dtype=np.int64)] = True
    out.extend(np.flatnonzero(~mask).tolist())
    return np.asarray(out, dtype=np.int64)


def page_weights(c: Corpus, qrows: list[int], rel_pages: list[set[int]]) -> np.ndarray:
    W = np.zeros(c.n_pages, dtype=np.float64)
    for qi in qrows:
        for p in rel_pages[qi]:
            W[p] += 1
    return W


def evidence_counts(ev: FastEval, state: np.ndarray, qrows: list[int],
                    pages_of_interest: set[int]) -> np.ndarray:
    """块的历史证据次数：历史题结果里落在关注页上的命中块各记一次。"""
    counts = np.zeros(ev.c.n, dtype=np.int32)
    for qi in qrows:
        for d in ev.top_chunks_of_pages(state, qi, pages_of_interest):
            counts[d] += 1
    return counts


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #

def wilcoxon_exact(diffs: list[float]) -> dict:
    """exact Wilcoxon signed-rank（零差剔除，average rank），返回单尾/双尾 p。

    单尾方向在 PROTOCOL §5 冻结（GREEDY > baseline）；n=5 时双尾 exact 下限 0.0625，
    单尾 0.03125 —— 这是离散下限，不是事后换检验。
    """
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return {'n': 0, 'w_plus': 0.0, 'p_one_sided': 1.0, 'p_two_sided': 1.0,
                'rank_biserial': 0.0}
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if d[i] > 0)
    # 以 2×rank 为整数的子集和 DP，统计 W+ 分布
    r2 = [int(round(2 * r)) for r in ranks]
    total = sum(r2)
    dp = [0] * (total + 1)
    dp[0] = 1
    for r in r2:
        for s in range(total, r - 1, -1):
            if dp[s - r]:
                dp[s] += dp[s - r]
    w_obs = int(round(2 * w_plus))
    cnt_ge = sum(dp[s] for s in range(w_obs, total + 1))
    cnt_le = sum(dp[s] for s in range(0, w_obs + 1))
    denom = 2 ** n
    p_hi = cnt_ge / denom
    p_lo = cnt_le / denom
    # 秩二列效应量（rank-biserial r）：正差秩和 - 负差秩和，再除以总秩和，∈[-1,1]
    w_minus = sum(ranks[i] for i in range(n) if d[i] < 0)
    denom_r = w_plus + w_minus
    r_rb = (w_plus - w_minus) / denom_r if denom_r else 0.0
    return {'n': n, 'w_plus': w_plus, 'p_one_sided': p_hi,
            'p_two_sided': min(1.0, 2 * min(p_hi, p_lo)), 'rank_biserial': r_rb}

def bootstrap_delta_stratified(before_reps: list[np.ndarray], after_reps: list[np.ndarray],
                               n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """以 **rep 为分析单位** 的分层 bootstrap 95% CI。

    两层：先对 5 个 rep 有放回重采样（尊重"重复是设计单位"），再在抽中的 rep 内按题配对重采样。
    返回 Δrecall(pp) 的 2.5/97.5 分位。用于和"池化 275 个观测"的 CI 并列比较（B-M2）。
    """
    rng = np.random.default_rng(seed)
    n_rep = len(before_reps)
    if n_rep == 0:
        return (float('nan'), float('nan'))
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        reps = rng.integers(0, n_rep, n_rep)
        acc = []
        for ri in reps:
            rb, ra = before_reps[int(ri)], after_reps[int(ri)]
            idx = rng.integers(0, len(rb), len(rb))
            acc.append(100.0 * (ra[idx].mean() - rb[idx].mean()))
        vals[b] = float(np.mean(acc))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def bootstrap_delta(before: np.ndarray, after: np.ndarray, n_boot: int = 1000,
                    seed: int = 0) -> tuple[float, float]:
    """Δrecall(pp) 的按题配对 bootstrap 95% CI。"""
    rng = np.random.default_rng(seed)
    n = len(before)
    if n == 0:
        return (float('nan'), float('nan'))
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = 100.0 * (after[idx].mean() - before[idx].mean())
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def bootstrap_recovery(r_before: np.ndarray, r_after: np.ndarray, r_full: np.ndarray,
                       n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """按题配对 bootstrap（同一题目下标同时取三个状态），返回 recovery 的 2.5/97.5 分位。"""
    rng = np.random.default_rng(seed)
    n = len(r_before)
    if n == 0:
        return (float('nan'), float('nan'))
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        rb, ra, rf = r_before[idx].mean(), r_after[idx].mean(), r_full[idx].mean()
        den = rf - rb
        vals[b] = (ra - rb) / den if den > 1e-12 else float('nan')
    return (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))
