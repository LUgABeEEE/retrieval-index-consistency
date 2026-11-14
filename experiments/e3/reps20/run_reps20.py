#!/usr/bin/env python3
"""E3 重复 5→20（t30）：同口径、同场景、600 s 单档；不重新嵌入（复用 E2 向量）。

产物：table.csv（20-rep 汇总 + 逐 rep）、results.json（统计与判据）。
设计见 PROTOCOL.md（先于结果冻结）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
E3 = HERE.parent
sys.path.insert(0, str(E3))
sys.path.insert(0, str(E3 / 'online-bandit'))
sys.path.insert(0, str(E3.parent / 'e2'))
sys.path.insert(0, str(E3.parent / 'hpc' / 'pylibs'))

import numpy as np  # noqa: E402

import e3_lib as L  # noqa: E402
import run_e3 as R  # noqa: E402
import run_online as O  # noqa: E402

ARMS = ['greedy-full', 'greedy-nolabel', 'online', 'fifo', 'lru', 'freq', 'random']
ARM_KEY = {'greedy-full': 'greedy', 'greedy-nolabel': 'greedy-nolabel',
           'fifo': 'fifo', 'lru': 'lru', 'freq': 'freq', 'random': 'random'}
BASELINES = ['fifo', 'lru', 'freq', 'random']


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def iqr(v):
    q1, q3 = np.percentile(v, [25, 75])
    return float(q1), float(q3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default=str(E3.parent / 'e2' / 'corpus-structural.jsonl.gz'))
    ap.add_argument('--vectors', default=str(E3.parent / 'e2' / 'vectors' / 'vectors.npy'))
    ap.add_argument('--qvecs', default=str(E3.parent / 'zzu-pilot' / 'qvecs.npy'))
    ap.add_argument('--gold', default=str(E3.parent / 'zzu-pilot' / 'gold.json'))
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--budget', type=int, default=600)
    ap.add_argument('--scenarios', default='full,hot,edit24')
    ap.add_argument('--seed', type=int, default=20261110)
    ap.add_argument('--out', default=str(HERE))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    scenarios = [s.strip() for s in args.scenarios.split(',') if s.strip()]
    t0 = time.time()

    gold, Q, usable, corpus, ev, rel, ages, change, sec = O.load_meta(args)
    allrows = np.arange(corpus.n, dtype=np.int64)
    full_q = ev.recall(allrows, usable).astype(np.float64)
    K = min(int(args.budget / sec), corpus.n)
    log(f'corpus={corpus.n} usable={len(usable)} reps={args.reps} budget={args.budget}s K={K}')

    per = {}   # (scenario, arm) -> list of (delta_pp, recall)
    for sc in scenarios:
        builder = R.make_builder(sc, corpus, ev, rel, usable)
        for r in range(args.reps):
            rng = np.random.default_rng(args.seed + r)
            perm = rng.permutation(len(usable))
            n_eval = len(usable) // 2 + 1
            eval_pos = sorted(perm[:n_eval].tolist())
            hist_pos = sorted(perm[n_eval:].tolist())
            eval_rows = [usable[i] for i in eval_pos]
            hist_rows = [usable[i] for i in hist_pos]
            debt, index0, _note = builder(hist_rows, rng)
            before_q = ev.recall(index0, usable).astype(np.float64)
            rb = before_q[eval_pos].mean()
            orders, _events = R.strategy_orders(corpus, ev, hist_rows, rel, rng,
                                                allrows, usable, ages, change)
            rank = np.empty(corpus.n, dtype=np.int64)
            for arm, key in ARM_KEY.items():
                rank[orders[key]] = np.arange(corpus.n)
                rest = debt[np.argsort(rank[debt])]
                state = np.sort(np.concatenate([index0, rest[:K]]))
                rec = float(ev.recall(state, eval_rows).mean())
                per.setdefault((sc, arm), []).append((100 * (rec - rb), rec))
            chosen = O.online_order(ev, corpus, debt, index0, hist_rows, K, change,
                                    O.PARAMS, prior_rows=hist_rows)
            state = np.sort(np.concatenate([index0, chosen]))
            rec = float(ev.recall(state, eval_rows).mean())
            per.setdefault((sc, 'online'), []).append((100 * (rec - rb), rec))
        log(f'{sc}: {args.reps} reps done  ({time.time()-t0:.0f}s)')

    table, results = [], {}
    for sc in scenarios:
        means = {a: float(np.mean([x[0] for x in per[(sc, a)]])) for a in ARMS}
        best = max(BASELINES, key=lambda a: means[a])
        for arm in ARMS:
            vals = np.array([x[0] for x in per[(sc, arm)]], dtype=np.float64)
            recs = np.array([x[1] for x in per[(sc, arm)]], dtype=np.float64)
            q1, q3 = iqr(vals)
            d_fifo = [per[(sc, arm)][i][0] - per[(sc, 'fifo')][i][0] for i in range(len(vals))]
            d_best = [per[(sc, arm)][i][0] - per[(sc, best)][i][0] for i in range(len(vals))]
            wf = L.wilcoxon_exact(d_fifo)
            wb = L.wilcoxon_exact(d_best)
            sd = float(np.std(d_fifo, ddof=1)) if len(d_fifo) > 1 else 0.0
            dz = float(np.mean(d_fifo) / sd) if sd > 1e-12 else float('nan')
            row = {'scenario': sc, 'arm': arm, 'budget_s': args.budget, 'n_reps': len(vals),
                   'delta_mean_pp': float(vals.mean()), 'delta_median_pp': float(np.median(vals)),
                   'delta_q1_pp': q1, 'delta_q3_pp': q3, 'delta_sd_pp': float(vals.std(ddof=1)),
                   'recall_mean': float(recs.mean()),
                   'p_vs_fifo_one_sided': wf['p_one_sided'], 'r_vs_fifo': wf['rank_biserial'],
                   'dz_vs_fifo': dz,
                   'p_vs_bestbaseline_one_sided': wb['p_one_sided'], 'r_vs_bestbaseline': wb['rank_biserial'],
                   'best_baseline': best,
                   'per_rep_delta_pp': [round(float(v), 4) for v in vals]}
            table.append(row)
    # 判据：full/600 greedy vs fifo
    g = next(r for r in table if r['scenario'] == 'full' and r['arm'] == 'greedy-full')
    crit = {'scenario': 'full', 'budget_s': args.budget, 'comparison': 'greedy-full vs fifo',
            'n_reps': args.reps, 'delta_mean_pp': g['delta_mean_pp'],
            'delta_median_pp': g['delta_median_pp'],
            'iqr_pp': [g['delta_q1_pp'], g['delta_q3_pp']],
            'p_one_sided': g['p_vs_fifo_one_sided'], 'r_rank_biserial': g['r_vs_fifo'],
            'dz': g['dz_vs_fifo'], 'best_baseline': g['best_baseline'],
            'criterion': 'p<0.01 and r>=0.5'}
    crit['robust'] = bool(crit['p_one_sided'] < 0.01 and crit['r_rank_biserial'] >= 0.5)
    # online hot 对比
    for comp in ('greedy-full', 'greedy-nolabel'):
        h_on = next(r for r in table if r['scenario'] == 'hot' and r['arm'] == 'online')
        h_c = next(r for r in table if r['scenario'] == 'hot' and r['arm'] == comp)
        d = [per[('hot', 'online')][i][0] - per[('hot', comp)][i][0] for i in range(args.reps)]
        wt = L.wilcoxon_exact(d)
        crit[f'hot_online_vs_{comp}'] = {'mean_gap_pp': float(np.mean(d)),
                                         'p_one_sided': wt['p_one_sided'],
                                         'r': wt['rank_biserial'],
                                         'ratio': (h_on['delta_mean_pp'] / h_c['delta_mean_pp']
                                                   if abs(h_c['delta_mean_pp']) > 1e-9 else float('nan'))}
    log('criterion: ' + json.dumps({k: crit[k] for k in
                                   ('delta_mean_pp', 'delta_median_pp', 'iqr_pp', 'p_one_sided',
                                    'r_rank_biserial', 'dz', 'robust')}, ensure_ascii=False))

    with open(out / 'table.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        for row in table:
            w.writerow({**row, 'per_rep_delta_pp': ' '.join(f'{x:.4f}' for x in row['per_rep_delta_pp'])})
    json.dump({'meta': {'corpus_n': corpus.n, 'reps': args.reps, 'budget_s': args.budget, 'K': K,
                        'scenarios': scenarios, 'seed': args.seed,
                        'elapsed_s': round(time.time() - t0, 1)},
               'criterion': crit, 'table': table},
              open(out / 'results.json', 'w'), ensure_ascii=False, indent=1)
    log(f'wrote {out}/table.csv, results.json ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
