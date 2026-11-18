#!/usr/bin/env python3
"""E3：固定预算下的修复调度（六策略 × 三档预算 × 5 次重复）。

设计见 `PROTOCOL.md` v1.1（先于正式数据冻结）。口径见 `experiments/METRICS.md` v2。

场景
  full   重分块后的全量欠账（主场景；R_before = 仅词法，R_full = 全量重建**参照**）
  hot    需求驱动的编辑积压：欠账页 = 事故前日志里历史查询 dense top-10 命中过的页（每 rep 重算）
  edit24 真实 24h 编辑欠账（manifest mtime）—— 诊断「修索引不一定涨召回」的非单调性

判据（captain 2026-11-10 批准，不得改）：600 s 上，GREEDY 比最佳基线
**+5 pp 绝对 recall@5**，且配对单尾 Wilcoxon p<0.05。归一化恢复率只作附加列，
只在「分母>0 且 R_after≤R_full」时给，否则 n/a。

用法
  python3 experiments/e3/run_e3.py --scenarios full,hot,edit24 --robust-reps 20
  python3 experiments/e3/run_e3.py --dry             # E1 旧 79,867 块，只验管线（R_full=0.486239）
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'e2'))
sys.path.insert(0, str(HERE.parent / 'hpc' / 'pylibs'))

import numpy as np  # noqa: E402

import e3_lib as E  # noqa: E402

STRATEGIES = ['fifo', 'lru', 'freq', 'random', 'greedy']
BASELINES = ['fifo', 'lru', 'freq', 'random']
ALL_STRATS = STRATEGIES + ['greedy-nolabel']      # no-label 变体（B-M1），不进判据
MANIFEST = '$KB_ROOT/state/manifest.json'
SNAP_TS = 1793969677.0        # 2026-11-06 20:54:37 +08:00（kb.sqlite.bak-20261106-205437）


def log(msg: str):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def load_inputs(args):
    gold, qs = E.load_gold(args.gold)
    Q = gold['questions']
    usable = [i for i, q in enumerate(Q) if q['qid'] in gold['qrels']]

    if args.dry:
        corpus = E.load_corpus_from_pickle(args.dry_base)
        vec_path = str(HERE.parent / 'zzu-pilot' / 'vecs.npy')
        tag = f'dry-{corpus.n}'
        rate_default = 294.0
    else:
        corpus = E.load_corpus(args.corpus)
        vec_path = args.vectors
        tag = f'full-{corpus.n}'
        rate_default = 296.5

    qv_all = np.load(args.qvecs).astype(np.float32)
    if qv_all.shape[0] != len(Q):
        raise SystemExit(f'qvecs {qv_all.shape} 与 gold.questions {len(Q)} 不匹配')
    V = np.load(vec_path, mmap_mode='r')
    if V.shape[0] != corpus.n or V.shape[1] != qv_all.shape[1]:
        raise SystemExit(f'vectors {V.shape} 与语料 {corpus.n} / 维度 {qv_all.shape[1]} 不匹配')

    rate = rate_default
    meta_p = Path(vec_path).parent / 'meta.json'
    if meta_p.exists():
        try:
            meta = json.load(open(meta_p))
            rate = float(meta.get('throughput_chunks_per_min') or rate)
            if not args.dry and meta.get('corpus', {}).get('sha256'):
                import hashlib
                h = hashlib.sha256()
                with open(args.corpus, 'rb') as fh:
                    for blk in iter(lambda: fh.read(1 << 20), b''):
                        h.update(blk)
                got = h.hexdigest()
                want = meta['corpus']['sha256']
                if got != want:
                    raise SystemExit(f'语料 sha256 与 meta.json 不一致：{got[:16]} != {want[:16]}'
                                     '（向量是对另一份语料算的，拒绝跑）')
                log(f'语料 sha256 与 meta.json 一致（{got[:16]}…）')
            log(f'吞吐来自 {meta_p.name}: {rate:.1f} 块/分（batch={meta.get("batch")}, '
                f'assemble={meta.get("assemble")}, model={meta.get("model")}）')
        except Exception as exc:                      # noqa: BLE001
            log(f'读 {meta_p} 失败（{exc}），回退 {rate:.1f} 块/分')
    sec = 60.0 / rate
    log(f'语料 {corpus.n} 块 / {corpus.n_pages} 页；向量 {V.shape}；单块 {sec:.4f}s')
    return gold, Q, usable, corpus, qv_all, V, tag, sec, vec_path


# --------------------------------------------------------------------------- #
# 欠账场景：返回 builder(hist_rows, rng) -> (debt, index0, note)
# --------------------------------------------------------------------------- #

def make_builder(scenario: str, corpus, ev, rel, usable):
    allrows = np.arange(corpus.n, dtype=np.int64)

    if scenario == 'full':
        return lambda hist_rows, rng: (allrows, np.zeros(0, dtype=np.int64),
                                       '重分块后全部块身份改变 → 稠密臂为空')
    if scenario == 'hot':
        def build(hist_rows, rng):
            pages = set()
            for qi in hist_rows:
                for p in ev.top_pages(allrows, qi)[:10]:
                    pages.add(p)
            debt = np.array(sorted(i for p in pages for i in corpus.page_chunks[p]), dtype=np.int64)
            mask = np.ones(corpus.n, dtype=bool); mask[debt] = False
            return debt, np.flatnonzero(mask), f'事故前日志中历史查询 dense top-10 命中过的 {len(pages)} 页成为欠账'
        return build
    if scenario == 'edit24':
        m = json.load(open(MANIFEST))['files']
        pid = {p: i for i, p in enumerate(corpus.page_names)}
        pages = [p for p, v in m.items()
                 if v.get('mtime') and SNAP_TS - 86400 <= v['mtime'] <= SNAP_TS and p in pid]
        debt = np.array(sorted(i for p in pages for i in corpus.page_chunks[pid[p]]), dtype=np.int64)
        mask = np.ones(corpus.n, dtype=bool); mask[debt] = False
        idx0 = np.flatnonzero(mask)
        note = f'manifest 显示快照前 24h 编辑过的 {len(pages)} 页 / {len(debt)} 块'
        return lambda hist_rows, rng: (debt, idx0, note)
    raise SystemExit(f'未知场景 {scenario}')


def manifest_age_rank(corpus) -> np.ndarray:
    """页年龄（快照时刻 - manifest mtime，天）→ [0,1] 秩，越老越接近 1；缺失填中位数。"""
    m = json.load(open(MANIFEST))['files']
    age = np.full(corpus.n_pages, np.nan, dtype=np.float64)
    for i, p in enumerate(corpus.page_names):
        v = m.get(p)
        if v and v.get('mtime'):
            age[i] = (SNAP_TS - float(v['mtime'])) / 86400.0
    med = np.nanmedian(age) if np.isfinite(age).any() else 0.0
    age = np.where(np.isfinite(age), age, med)
    lo, hi = float(age.min()), float(age.max())
    return (age - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(age)


def change_proxy(corpus) -> np.ndarray:
    """内容变更幅度代理：块字符数 / 该页中位块字符数（旧文本不可得，用体量代理；无标注）。"""
    lens = np.array([len(t) for t in corpus.texts], dtype=np.float64)
    med = np.array([np.median(lens[ch]) if ch else 1.0 for ch in corpus.page_chunks])
    return lens / np.maximum(med[corpus.page_ids], 1.0)


def strategy_orders(corpus, ev, hist_rows, rel, rng, allrows, usable, ages, change):
    events = [(int(ts), ev.top_pages(allrows, qi))
              for ts, qi in zip(rng.permutation(len(hist_rows)), hist_rows)]
    W = E.page_weights(corpus, hist_rows, rel)
    pages_hit = {p for _ts, pages in events for p in pages} | {p for p in range(corpus.n_pages) if W[p] > 0}
    evidence = E.evidence_counts(ev, allrows, hist_rows, pages_hit)
    # B-M1：无标注变体。只用可观测信号：
    #   命中频次 hits(p)（历史查询返回 top-10 的页计数）
    #   页年龄 age(p)（manifest mtime，越老权重略高）
    #   内容变更幅度 proxy（块长/页中位块长）
    # 完全不使用 gold 标注；同样走 sqrt 凹边际收益的贪心。
    hits = {}
    for _ts, pages in events:
        for p in pages:
            hits[p] = hits.get(p, 0) + 1
    W_nl = np.array([hits.get(p, 0) * (1.0 + 0.5 * float(ages[p]))
                     for p in range(corpus.n_pages)], dtype=np.float64)
    return {
        'fifo': E.order_fifo(corpus),
        'random': E.order_random(corpus, rng),
        'freq': E.order_by_pages(corpus, E.page_rank_from_events(corpus, events, 'freq'), evidence),
        'lru': E.order_by_pages(corpus, E.page_rank_from_events(corpus, events, 'lru'), evidence),
        'greedy': E.order_greedy(corpus, W, evidence),
        'greedy-oracle': E.order_greedy(corpus, E.page_weights(corpus, usable, rel), evidence),
        'greedy-nolabel': E.order_greedy(corpus, W_nl, None, chunk_key=change),
    }, events


def page_disjoint_split(usable, Q, qrels, rng=None):
    """按 gold-page 连通分量把题分成两半：任何 page 只出现在一半里。

    为什么需要这个（2026-11-18 混元评审指出）：原来的划分按 query 随机切，
    但 109 个问题只对应 48 个 gold page（平均每页 2.3 题），随机切之后约 56%
    的历史侧 gold page 同时出现在评测侧，历史标签等于直接指向评测答案页，
    会给调度类方法一个不真实的优势。这里用并查集把共享 page 的题聚成一组，
    整组分配，保证 page 层面零重叠。
    """
    n = len(usable)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    page_first = {}
    for pos, qi in enumerate(usable):
        for g in qrels.get(Q[qi]['qid'], []):
            pth = g.get('path') if isinstance(g, dict) else None
            if not pth:
                continue
            if pth in page_first:
                union(pos, page_first[pth])
            else:
                page_first[pth] = pos

    comps = {}
    for pos in range(n):
        comps.setdefault(find(pos), []).append(pos)

    # 大分量先分配，贪心让两半题数接近
    eval_set, hist_set = [], []
    for c in sorted(comps.values(), key=len, reverse=True):
        if len(eval_set) <= len(hist_set):
            eval_set.extend(c)
        else:
            hist_set.extend(c)
    return sorted(eval_set), sorted(hist_set)


def run_scenario(name, builder, corpus, ev, usable, rel, budgets_k, reps, seed, ages, change,
                 Q=None, qrels=None, split='query'):
    allrows = np.arange(corpus.n, dtype=np.int64)
    full_q = ev.recall(allrows, usable).astype(np.float64)
    canon_full = float(full_q.mean())          # 109 题口径的全量重建参照
    rows, details = [], []
    rec = {}
    for r in reps:
        rng = np.random.default_rng(seed + r)
        if split == 'page' and Q is not None and qrels is not None:
            eval_pos, hist_pos = page_disjoint_split(usable, Q, qrels, rng)
        else:
            perm = rng.permutation(len(usable))
            n_eval = len(usable) // 2 + 1
            eval_pos = sorted(perm[:n_eval].tolist())
            hist_pos = sorted(perm[n_eval:].tolist())
        eval_rows = [usable[i] for i in eval_pos]
        hist_rows = [usable[i] for i in hist_pos]
        debt, index0, note = builder(hist_rows, rng)
        before_q = ev.recall(index0, usable).astype(np.float64)
        canon_before = float(before_q.mean())   # 109 题口径的 R_before（不随切分变）
        rb, rf = before_q[eval_pos].mean(), full_q[eval_pos].mean()
        den = rf - rb
        orders, _events = strategy_orders(corpus, ev, hist_rows, rel, rng, allrows, usable, ages, change)
        log(f'[{name}] rep{r}: {note}；debt={len(debt)}（{100*len(debt)/corpus.n:.1f}%）'
            f' index0={len(index0)}；R_before={rb:.4f} R_full={rf:.4f} den={den:+.4f}')

        rank = np.empty(corpus.n, dtype=np.int64)
        for strat in ALL_STRATS + ['greedy-oracle']:
            rank[orders[strat]] = np.arange(corpus.n)
            rest = debt[np.argsort(rank[debt])]
            for b in budgets_k:
                k = min(budgets_k[b], len(debt))
                state = np.sort(np.concatenate([index0, rest[:k]]))
                q_rec = ev.recall(state, eval_rows).astype(np.float64)
                ra = q_rec.mean()
                rows.append({'scenario': name, 'rep': r, 'budget_s': b, 'strategy': strat,
                             'n_embeds': int(k), 'debt': len(debt),
                             'recovery': (ra - rb) / den if den > 1e-12 else None,
                             'recall_after_eval': ra, 'recall_before_109': canon_before,
                             'recall_full_109': canon_full,
                             'delta_recall_pp': 100 * (ra - rb)})
                d = rec.setdefault((strat, b), {'recovery': [], 'after': [], 'before': [], 'full': []})
                d['recovery'].append((ra - rb) / den if den > 1e-12 else float('nan'))
                d['after'].append(q_rec)
                d['before'].append(before_q[eval_pos])
                d['full'].append(full_q[eval_pos])
        details.append({'rep': r, 'eval_rows': eval_rows, 'hist_rows': hist_rows,
                        'R_before_eval': rb, 'R_full_eval': rf,
                        'R_before_109': canon_before, 'R_full_109': canon_full,
                        'debt_chunks': int(len(debt)),
                        'debt_pages': int(len({int(corpus.page_ids[i]) for i in debt}))})

    table = []
    for strat in ALL_STRATS + ['greedy-oracle']:
        for b in budgets_k:
            d = rec[(strat, b)]
            rb = np.concatenate(d['before']); ra = np.concatenate(d['after'])
            rf = np.concatenate(d['full'])
            dlo, dhi = E.bootstrap_delta(rb, ra, n_boot=1000, seed=7 + b)
            slo, shi = E.bootstrap_delta_stratified(d['before'], d['after'],
                                                   n_boot=1000, seed=13 + b)
            den_pos = all((float(np.mean(d['full'][i])) - float(np.mean(d['before'][i]))) > 0
                          for i in range(len(reps)))
            within = all(float(np.mean(d['after'][i])) <= float(np.mean(d['full'][i])) + 1e-12
                         for i in range(len(reps)))
            valid = den_pos and within
            rlo, rhi = (E.bootstrap_recovery(rb, ra, rf, n_boot=1000, seed=11 + b)
                        if valid else (float('nan'), float('nan')))
            note = ('ok' if valid else
                    ('n/a: R_after > R_full（非单调）' if den_pos else
                     ('n/a: 分母 ≤ 0' if within else 'n/a: 分母 ≤ 0 且 R_after > R_full')))
            delta_reps = [100.0 * (float(np.mean(d['after'][i])) - float(np.mean(d['before'][i])))
                          for i in range(len(reps))]
            table.append({'scenario': name, 'strategy': strat, 'budget_s': b,
                          'n_embeds': int(min(budgets_k[b], details[0]['debt_chunks'])),
                          'recall_before_109': float(np.mean([x['R_before_109'] for x in details])),
                          'recall_full_109': canon_full,
                          'recall_before_eval': float(rb.mean()),
                          'recall_after_eval': float(ra.mean()),
                          'delta_recall_pp': float(100 * (ra.mean() - rb.mean())),
                          'delta_ci_lo_pp': dlo, 'delta_ci_hi_pp': dhi,
                          'ci_stratified_lo_pp': slo, 'ci_stratified_hi_pp': shi,
                          'delta_min_pp': float(np.min(delta_reps)),
                          'delta_max_pp': float(np.max(delta_reps)),
                          'delta_range_pp': float(np.max(delta_reps) - np.min(delta_reps)),
                          'recovery_mean': float(np.mean(d['recovery'])) if valid else None,
                          'recovery_median': float(np.median(d['recovery'])) if valid else None,
                          'recovery_sd': float(np.std(d['recovery'], ddof=1)) if valid else None,
                          'recovery_ci_lo': rlo, 'recovery_ci_hi': rhi,
                          'recovery_n_valid': int(sum(1 for x in d['recovery'] if x == x and x <= 1.0)),
                          'recovery_note': note,
                          'per_rep_delta_pp': [round(float(x), 4) for x in delta_reps],
                          'per_rep_recovery': [round(float(x), 6) if x == x else None
                                               for x in d['recovery']]})
    for b in budgets_k:
        cb = float(np.mean([x['R_before_109'] for x in details]))
        table.append({'scenario': name, 'strategy': 'full', 'budget_s': b,
                      'n_embeds': corpus.n,
                      'recall_before_109': cb, 'recall_full_109': canon_full,
                      'recall_before_eval': cb,
                      'recall_after_eval': canon_full,
                      'delta_recall_pp': float(100 * (canon_full - cb)),
                      'delta_ci_lo_pp': None, 'delta_ci_hi_pp': None,
                      'ci_stratified_lo_pp': None, 'ci_stratified_hi_pp': None,
                      'delta_min_pp': None, 'delta_max_pp': None, 'delta_range_pp': None,
                      'recovery_mean': None, 'recovery_median': None, 'recovery_sd': None,
                      'recovery_ci_lo': None, 'recovery_ci_hi': None,
                      'recovery_n_valid': 0, 'recovery_note': 'reference',
                      'per_rep_delta_pp': [], 'per_rep_recovery': []})

    pb = 600 if 600 in budgets_k else next(iter(budgets_k))
    bmap = {(row['strategy']): row for row in table if row['budget_s'] == pb}
    base_means = {s: bmap[s]['delta_recall_pp'] for s in BASELINES}
    best_base = max(base_means, key=base_means.get)
    g = bmap['greedy']
    diffs_best = [g['per_rep_delta_pp'][i] - bmap[best_base]['per_rep_delta_pp'][i]
                  for i in range(len(reps))]
    wt_best = E.wilcoxon_exact(diffs_best)
    stats = {'scenario': name, 'primary_budget_s': pb, 'best_baseline': best_base,
             'greedy_delta_pp': g['delta_recall_pp'],
             'best_baseline_delta_pp': base_means[best_base],
             'gap_pp': g['delta_recall_pp'] - base_means[best_base],
             'greedy_ci_pp': [g['delta_ci_lo_pp'], g['delta_ci_hi_pp']],
             'baseline_delta_pp': base_means,
             'vs_best': {'wilcoxon_one_sided': wt_best['p_one_sided'],
                         'wilcoxon_two_sided': wt_best['p_two_sided'],
                         'rank_biserial': wt_best['rank_biserial'],
                         'n_pairs': wt_best['n'],
                         'diffs_pp': [round(float(x), 4) for x in diffs_best]},
             'deltas': {}}
    for s in BASELINES + ['greedy-oracle']:
        diffs = [g['per_rep_delta_pp'][i] - bmap[s]['per_rep_delta_pp'][i] for i in range(len(reps))]
        wt = E.wilcoxon_exact(diffs)
        stats['deltas'][s] = {'gap_pp': float(np.mean(diffs)),
                              'one_sided': wt['p_one_sided'], 'two_sided': wt['p_two_sided'],
                              'rank_biserial': wt['rank_biserial']}
    criterion = (stats['gap_pp'] >= 5.0 and stats['vs_best']['wilcoxon_one_sided'] < 0.05)
    stats['criterion_pass'] = bool(criterion)
    log(f'[{name}] 600s: greedy Δrecall={g["delta_recall_pp"]:+.2f}pp '
        f'best={best_base}({base_means[best_base]:+.2f}pp) gap={stats["gap_pp"]:+.2f}pp '
        f'p1={stats["vs_best"]["wilcoxon_one_sided"]:.4f} → {"PASS" if criterion else "FAIL"}')
    return rows, table, stats, details


def run_robustness(builder, corpus, ev, usable, rel, K, n_reps, seed, ages, change):
    """额外 n_reps 次随机切分（只跑一个预算），返回 GREEDY vs 各基线的 Δrecall(pp) 与单尾 Wilcoxon。"""
    allrows = np.arange(corpus.n, dtype=np.int64)
    recs = {s: [] for s in ALL_STRATS + ['greedy-oracle']}
    for rr in range(n_reps):
        rng = np.random.default_rng(seed + rr)
        perm = rng.permutation(len(usable))
        n_eval = len(usable) // 2 + 1
        eval_pos = sorted(perm[:n_eval].tolist()); hist_pos = sorted(perm[n_eval:].tolist())
        eval_rows = [usable[i] for i in eval_pos]; hist_rows = [usable[i] for i in hist_pos]
        debt, index0, _note = builder(hist_rows, rng)
        before_q = ev.recall(index0, usable).astype(np.float64)
        rb = before_q[eval_pos].mean()
        orders, _events = strategy_orders(corpus, ev, hist_rows, rel, rng, allrows, usable, ages, change)
        debt_set = set(int(x) for x in debt)
        for s in recs:
            keep = np.array([d for d in orders[s].tolist() if d in debt_set][:K], dtype=np.int64)
            state = np.sort(np.concatenate([index0, keep]))
            recs[s].append(100.0 * (ev.recall(state, eval_rows).mean() - rb))
    out = {}
    for s in recs:
        diffs = [recs['greedy'][i] - recs[s][i] for i in range(len(recs['greedy']))]
        wt = E.wilcoxon_exact(diffs)
        out[s] = {'mean_delta_pp': float(np.mean(recs[s])), 'greedy_gap_pp': float(np.mean(diffs)),
                  'one_sided': wt['p_one_sided'], 'two_sided': wt['p_two_sided'],
                  'rank_biserial': wt['rank_biserial'], 'n_pairs': wt['n']}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true', help='用 zzu-pilot 旧 79,867 块干跑（只验管线）')
    ap.add_argument('--dry-base', default=str(HERE.parent / 'e1' / 'cache' / 'base.pkl'))
    ap.add_argument('--corpus', default=str(HERE.parent / 'e2' / 'corpus-structural.jsonl.gz'))
    ap.add_argument('--vectors', default=str(HERE.parent / 'e2' / 'vectors' / 'vectors.npy'))
    ap.add_argument('--qvecs', default=str(HERE.parent / 'zzu-pilot' / 'qvecs.npy'))
    ap.add_argument('--gold', default=str(HERE.parent / 'zzu-pilot' / 'gold.json'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--cache', default=str(HERE / 'cache'))
    ap.add_argument('--split', default='query', choices=['query', 'page'],
                    help='query=shuffle queries; page=merge queries sharing a gold page (zero overlap)')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--robust-reps', type=int, default=0)
    ap.add_argument('--budgets', default='300,600,1200')
    ap.add_argument('--scenarios', default='full,hot,edit24')
    ap.add_argument('--seed', type=int, default=20261110)
    args = ap.parse_args()

    # 干跑产物写 out-dry/，避免和正式产物混淆
    args.out = args.out or str(HERE / ('out-dry' if args.dry else 'out'))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    budgets = [int(x) for x in args.budgets.split(',')]
    scenarios = [s.strip() for s in args.scenarios.split(',') if s.strip()]
    t0 = time.time()

    gold, Q, usable, corpus, qv_all, V, tag, sec, vec_used = load_inputs(args)
    log('页面相关性（一次算好）…')
    rel = E.page_relevant_pages(corpus, gold, Q)
    bm25_idx = E.build_bm25_topk(corpus, Q, cache / f'bm25_{tag}_d{E.DEPTH}.npz', tag=tag)
    vec_path = str(V.filename) if hasattr(V, 'filename') else args.vectors
    S = E.build_dense_scores(qv_all, V, cache / f'dense_{tag}_{E.file_sig(vec_path)}.npy', tag=tag)
    ages = manifest_age_rank(corpus)          # B-M1 无标注变体用的页年龄
    change = change_proxy(corpus)             # B-M1 内容变更幅度代理（块长/页中位）
    corpus.drop_texts()
    ev = E.FastEval(S, bm25_idx, corpus, rel)
    if args.dry:
        rfull = ev.recall(np.arange(corpus.n), usable).mean()
        want = 0.486239
        if abs(rfull - want) < 2e-6:
            log(f'✅ 干跑 R_full={rfull:.6f} 与 METRICS v2 基线 {want} 一致（口径复现通过）')
        else:
            log(f'⚠️ 干跑 R_full={rfull:.6f} ≠ {want}（口径/数据变了，先查清再跑正式）')
    budgets_k = {b: min(int(b / sec), corpus.n) for b in budgets}
    log('预算→块数: ' + ', '.join(f'{b}s→{k}' for b, k in budgets_k.items()))

    all_rows, all_table, all_stats, all_details = [], [], {}, {}
    for sc in scenarios:
        builder = make_builder(sc, corpus, ev, rel, usable)
        rows, table, stats, details = run_scenario(
            sc, builder, corpus, ev, usable, rel, budgets_k, list(range(args.reps)), args.seed,
            ages, change, Q=Q, qrels=gold['qrels'], split=args.split)
        all_rows += rows; all_table += table; all_stats[sc] = stats; all_details[sc] = details
        if args.robust_reps:
            log(f'[{sc}] 鲁棒性：额外 {args.robust_reps} 次切分（只 {stats["primary_budget_s"]}s）…')
            rob = run_robustness(builder, corpus, ev, usable, rel,
                                 budgets_k[stats['primary_budget_s']], args.robust_reps,
                                 args.seed + 1000, ages, change)
            all_stats[sc]['robustness'] = rob
            log('  鲁棒性 gap: ' + ', '.join(f'{s}:{v["greedy_gap_pp"]:+.1f}pp(p={v["one_sided"]:.4f})'
                                            for s, v in rob.items()))

    def write_table(path, rows):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['scenario', 'strategy', 'budget_s', 'n_embeds',
                                              'recall_before_109', 'recall_full_109',
                                              'recall_before_eval', 'recall_after_eval',
                                              'delta_recall_pp',
                                              'delta_ci_lo_pp', 'delta_ci_hi_pp',
                                              'ci_stratified_lo_pp', 'ci_stratified_hi_pp',
                                              'delta_min_pp', 'delta_max_pp', 'delta_range_pp',
                                              'recovery_mean', 'recovery_median', 'recovery_sd',
                                              'recovery_ci_lo', 'recovery_ci_hi', 'recovery_n_valid',
                                              'recovery_note', 'per_rep_delta_pp', 'per_rep_recovery'])
            w.writeheader()
            for row in rows:
                def fr(x):                                       # None / NaN → n/a
                    return 'n/a' if x is None or (isinstance(x, float) and x != x) else x
                w.writerow({**row,
                            'recovery_mean': fr(row['recovery_mean']),
                            'recovery_median': fr(row['recovery_median']),
                            'recovery_sd': fr(row['recovery_sd']),
                            'recovery_ci_lo': fr(row['recovery_ci_lo']),
                            'recovery_ci_hi': fr(row['recovery_ci_hi']),
                            'delta_ci_lo_pp': fr(row['delta_ci_lo_pp']),
                            'delta_ci_hi_pp': fr(row['delta_ci_hi_pp']),
                            'per_rep_delta_pp': ' '.join(f'{x:.4f}' for x in row['per_rep_delta_pp']),
                            'per_rep_recovery': ' '.join(
                                ('n/a' if x is None else f'{x:.6f}') for x in row['per_rep_recovery'])})

    primary = scenarios[0]
    write_table(out / 'table3.csv', [r for r in all_table if r['scenario'] == primary])
    for sc in scenarios[1:]:
        write_table(out / f'table3_{sc}.csv', [r for r in all_table if r['scenario'] == sc])
    with open(out / 'fig4.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['scenario', 'rep', 'budget_s', 'strategy', 'n_embeds',
                                          'debt', 'recovery', 'recall_after_eval',
                                          'recall_before_109', 'recall_full_109', 'delta_recall_pp'])
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    # ---------------- B-M1 / B-M2 专项 CSV（不改已有数字，只增补分析） ----------------
    def write_special():
        # table3_nolabel.csv：主场景 full，三个预算；贪心 vs 全标注贪心 vs 最佳基线
        rows = [r for r in all_table if r['scenario'] == 'full']
        by = {(r['budget_s'], r['strategy']): r for r in rows}
        with open(out / 'table3_nolabel.csv', 'w', newline='', encoding='utf-8') as f:
            fields = ['strategy', 'budget_s', 'n_embeds', 'delta_recall_pp',
                      'ci_pooled_lo_pp', 'ci_pooled_hi_pp',
                      'ci_stratified_lo_pp', 'ci_stratified_hi_pp',
                      'delta_min_pp', 'delta_max_pp', 'delta_range_pp',
                      'per_rep_delta_pp', 'gap_vs_greedy_pp', 'gap_vs_best_baseline_pp',
                      'wilcoxon_vs_greedy_one_sided', 'wilcoxon_vs_best_baseline_one_sided', 'n_pairs']
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for b in budgets:
                brows = [r for r in rows if r['budget_s'] == b]
                bmap = {r['strategy']: r for r in brows}
                best = max(BASELINES, key=lambda s: bmap[s]['delta_recall_pp']) if bmap else None
                g = bmap['greedy']['per_rep_delta_pp']
                for s in ['greedy', 'greedy-nolabel', 'fifo', 'lru', 'freq', 'random', 'full']:
                    r = bmap.get(s)
                    if not r:
                        continue
                    pr = r['per_rep_delta_pp']
                    if s in ('full',):
                        gap_g = gap_b = p_g = p_b = 'n/a'
                    else:
                        gap_g = r['delta_recall_pp'] - bmap['greedy']['delta_recall_pp']
                        gap_b = r['delta_recall_pp'] - bmap[best]['delta_recall_pp']
                        dg = [g[i] - pr[i] for i in range(len(pr))]
                        db = [pr[i] - bmap[best]['per_rep_delta_pp'][i] for i in range(len(pr))]
                        p_g = E.wilcoxon_exact(dg)['p_one_sided']
                        p_b = E.wilcoxon_exact(db)['p_one_sided']
                    w.writerow({'strategy': s, 'budget_s': b, 'n_embeds': r['n_embeds'],
                                'delta_recall_pp': r['delta_recall_pp'],
                                'ci_pooled_lo_pp': r['delta_ci_lo_pp'], 'ci_pooled_hi_pp': r['delta_ci_hi_pp'],
                                'ci_stratified_lo_pp': r['ci_stratified_lo_pp'],
                                'ci_stratified_hi_pp': r['ci_stratified_hi_pp'],
                                'delta_min_pp': r['delta_min_pp'], 'delta_max_pp': r['delta_max_pp'],
                                'delta_range_pp': r['delta_range_pp'],
                                'per_rep_delta_pp': ' '.join(f'{x:.4f}' for x in pr),
                                'gap_vs_greedy_pp': gap_g, 'gap_vs_best_baseline_pp': gap_b,
                                'wilcoxon_vs_greedy_one_sided': p_g,
                                'wilcoxon_vs_best_baseline_one_sided': p_b,
                                'n_pairs': len(pr)})
        # ci_units.csv：两种 CI 分析单位并列（主场景 600 s，全部策略）
        with open(out / 'ci_units.csv', 'w', newline='', encoding='utf-8') as f:
            fields = ['scenario', 'strategy', 'budget_s', 'n_reps', 'n_obs',
                      'delta_mean_pp', 'delta_min_pp', 'delta_max_pp', 'delta_range_pp',
                      'ci_pooled_lo_pp', 'ci_pooled_hi_pp',
                      'ci_stratified_lo_pp', 'ci_stratified_hi_pp',
                      'per_rep_delta_pp', 'min_single_in_pooled', 'min_single_in_stratified']
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for r in all_table:
                if r['budget_s'] != 600 or r['strategy'] == 'full':
                    continue
                mn = r['delta_min_pp']
                w.writerow({'scenario': r['scenario'], 'strategy': r['strategy'], 'budget_s': 600,
                            'n_reps': len(r['per_rep_delta_pp']),
                            'n_obs': len(r['per_rep_delta_pp']) * (109 // 2 + 1),
                            'delta_mean_pp': r['delta_recall_pp'], 'delta_min_pp': mn,
                            'delta_max_pp': r['delta_max_pp'], 'delta_range_pp': r['delta_range_pp'],
                            'ci_pooled_lo_pp': r['delta_ci_lo_pp'], 'ci_pooled_hi_pp': r['delta_ci_hi_pp'],
                            'ci_stratified_lo_pp': r['ci_stratified_lo_pp'],
                            'ci_stratified_hi_pp': r['ci_stratified_hi_pp'],
                            'per_rep_delta_pp': ' '.join(f'{x:.4f}' for x in r['per_rep_delta_pp']),
                            'min_single_in_pooled': bool(r['delta_ci_lo_pp'] <= mn <= r['delta_ci_hi_pp']),
                            'min_single_in_stratified': bool(r['ci_stratified_lo_pp'] <= mn <= r['ci_stratified_hi_pp'])})
    write_special()

    json.dump({'meta': {'corpus_n': corpus.n, 'corpus_pages': corpus.n_pages, 'tag': tag,
                        'sec_per_chunk': sec, 'budgets_k': budgets_k, 'reps': args.reps,
                        'seed': args.seed, 'dry': args.dry, 'scenarios': scenarios,
                        'vectors': str(vec_used), 'corpus': str(args.corpus)},
               'stats': all_stats, 'table': all_table, 'reps': all_details},
              open(out / 'results.json', 'w'), ensure_ascii=False, indent=1)
    log(f'写出 {out}/table3.csv, fig4.csv, results.json  总耗时 {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
