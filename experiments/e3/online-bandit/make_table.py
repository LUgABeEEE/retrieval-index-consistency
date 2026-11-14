#!/usr/bin/env python3
"""生成两张表：
  table.csv        —— 论文用：三场景 × 关键臂（600 s）+ 11 个超参设置的最优值
  table_detail.csv —— 全量长表：scenario × budget × arm（含逐 rep）
"""
from __future__ import annotations
import csv, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ['online', 'greedy-full', 'greedy-nolabel', 'fifo', 'lru', 'random']


def main():
    res = json.load(open(HERE / 'results.json', encoding='utf-8'))
    table, rows = res['table'], res['rows']
    # 长表
    with open(HERE / 'table_detail.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['scenario', 'budget_s', 'arm', 'recall_mean',
                                          'delta_pp_mean', 'delta_pp_median', 'delta_pp_sd',
                                          'recovery_mean', 'recovery_note', 'per_rep_delta_pp'])
        w.writeheader()
        for r in table:
            w.writerow({**r,
                        'recovery_mean': 'n/a' if r['recovery_mean'] is None else r['recovery_mean'],
                        'per_rep_delta_pp': ' '.join(f'{x:.4f}' for x in r['per_rep_delta_pp'])})
    # 超参/机制设置
    grids = []
    for gp in sorted(HERE.glob('grid/*/results.json')):
        g = json.load(open(gp, encoding='utf-8'))['criterion']
        grids.append({'setting': gp.parent.name, **g})
    hyper = [g for g in grids if not g['params'].get('observe_full')][:11]   # α×c 6 + γ/R 5
    best = max(hyper, key=lambda g: g['gain_pp']['online']) if hyper else None
    best_o = (max(grids, key=lambda g: g['gain_pp']['online']) if grids else None)

    # 论文表：600 s，三场景 × 关键臂 + best-of-11
    fields = ['scenario', 'budget_s', 'online_pp', 'greedy_full_pp', 'greedy_nolabel_pp',
              'fifo_pp', 'lru_pp', 'random_pp',
              'online_recall', 'greedy_full_recall', 'greedy_nolabel_recall',
              'online_over_full_ratio', 'n_reps', 'per_rep_online_pp',
              'best_grid_setting', 'best_grid_online_pp', 'best_grid_ratio']
    with open(HERE / 'table.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sc in ('full', 'hot', 'edit24'):
            g = {a: next(t for t in table if t['scenario'] == sc and t['budget_s'] == 600 and t['arm'] == a)
                 for a in ARMS}
            ratio = (g['online']['delta_pp_mean'] / g['greedy-full']['delta_pp_mean']
                     if abs(g['greedy-full']['delta_pp_mean']) > 1e-9 else float('nan'))
            row = {'scenario': sc, 'budget_s': 600,
                   'online_pp': round(g['online']['delta_pp_mean'], 3),
                   'greedy_full_pp': round(g['greedy-full']['delta_pp_mean'], 3),
                   'greedy_nolabel_pp': round(g['greedy-nolabel']['delta_pp_mean'], 3),
                   'fifo_pp': round(g['fifo']['delta_pp_mean'], 3),
                   'lru_pp': round(g['lru']['delta_pp_mean'], 3),
                   'random_pp': round(g['random']['delta_pp_mean'], 3),
                   'online_recall': round(g['online']['recall_mean'], 6),
                   'greedy_full_recall': round(g['greedy-full']['recall_mean'], 6),
                   'greedy_nolabel_recall': round(g['greedy-nolabel']['recall_mean'], 6),
                   'online_over_full_ratio': round(ratio, 3),
                   'n_reps': len(g['online']['per_rep_delta_pp']),
                   'per_rep_online_pp': ' '.join(f'{x:.2f}' for x in g['online']['per_rep_delta_pp']),
                   'best_grid_setting': 'n/a', 'best_grid_online_pp': 'n/a', 'best_grid_ratio': 'n/a'}
            if sc == 'full' and best:
                row.update(best_grid_setting=best['setting'],
                           best_grid_online_pp=round(best['gain_pp']['online'], 3),
                           best_grid_ratio=round(best['online_over_full_ratio'], 3))
            w.writerow(row)
    print(f'wrote table.csv ({3} rows) + table_detail.csv ({len(table)} rows); '
          f'best-of-{len(hyper)} hyper = {best["setting"] if best else None} '
          f'({best["gain_pp"]["online"]:+.2f}pp)' if best else '')


if __name__ == '__main__':
    main()
