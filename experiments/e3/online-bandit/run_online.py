#!/usr/bin/env python3
"""A1：在线 restless-bandit（Whittle 指数式）修复调度 —— 实现与评估（t26）。

设计（先写死，见 notes.md §1；不读 gold）
  * arm = 页；奖励只有在查询把它检索进 top-10 时才兑现（有排名权重 1/(rank+1)）。
  * 每轮（round）：先服务一批历史查询（只用日志，观察 top-10 页），再按优先级排序、
    把预算内的欠账块嵌入；共 R=6 轮。
  * Whittle 指数式优先级（只用在线观测量）：
        rate_p = (Σ_window 1/(rank+1) + α) / (|window| + 2α)          # 观测命中率（滑动窗口 W=20）
        ucb_p  = 1 + c·sqrt(log(t+1)/(n_p+1))                          # 探索项（restless bandit 的乐观项）
        stale_p= 1 + β·(round − last_repair_p)/R                       # 距上次重嵌的时长
        index_p= rate_p · ucb_p · stale_p
    页内块序 = 内容变更幅度代理（块长/页中位）降序；同页边际收益按 sqrt(n/N_p) 递减
    （复用 E3 的 `order_greedy` 堆实现）。
  * 参数：R=6, W=20, α=0.5, c=0.5, β=0.5。全部在跑之前写死。
评估
  同 E3：131,028 块、109 题、场景 full/hot/edit24、预算 300/600/1200s、5 次重复。
  对比臂：online / full-label greedy / no-label 代理 / FIFO / LRU / random。
  指标：评测半边（55 题）上的绝对 recall@5 与 Δrecall(pp)；归一化恢复率仅在分母>0 且
  R_after≤R_full 时给；单尾 Wilcoxon（online vs 各臂）。
判据（先写死）：online 优于 no-label 代理（均值更高且单尾 p<0.05）且达到 full-label 贪心
  收益的 ≥50% ⇒「在线变体有实用价值」；与 no-label 无差异 ⇒「在线估计不足以替代标注信号」。

产物：table.csv / notes.md（本脚本只写 results.json + fig 数据，notes 由 make_notes 生成）。
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
E3 = HERE.parent
sys.path.insert(0, str(E3))
sys.path.insert(0, str(E3.parent / 'e2'))
sys.path.insert(0, str(E3.parent / 'hpc' / 'pylibs'))

import numpy as np  # noqa: E402

import e3_lib as L  # noqa: E402
import run_e3 as R  # noqa: E402

PARAMS = dict(rounds=6, window=20, alpha=0.5, ucb_c=0.5, beta=0.5, gamma=0.95)
ARMS = ['online', 'greedy-full', 'greedy-nolabel', 'fifo', 'lru', 'random']


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def load_meta(args):
    gold, qs = L.load_gold(args.gold)
    Q = gold['questions']
    usable = [i for i, q in enumerate(Q) if q['qid'] in gold['qrels']]
    corpus = L.load_corpus(args.corpus)
    qv_all = np.load(args.qvecs).astype(np.float32)
    V = np.load(args.vectors, mmap_mode='r')
    if V.shape[0] != corpus.n:
        raise SystemExit('vectors/corpus mismatch')
    rate = 309.08
    mp = Path(args.vectors).parent / 'meta.json'
    if mp.exists():
        rate = float(json.load(open(mp)).get('throughput_chunks_per_min') or rate)
    sec = 60.0 / rate
    rel = L.page_relevant_pages(corpus, gold, Q)
    tag = f'full-{corpus.n}'
    bm = L.build_bm25_topk(corpus, Q, E3 / 'cache' / f'bm25_{tag}_d50.npz', tag=tag)
    S = L.build_dense_scores(qv_all, V, E3 / 'cache' / f'dense_{tag}_{L.file_sig(args.vectors)}.npy', tag=tag)
    ages = R.manifest_age_rank(corpus)
    change = R.change_proxy(corpus)
    corpus.drop_texts()
    ev = L.FastEval(S, bm, corpus, rel)
    return gold, Q, usable, corpus, ev, rel, ages, change, sec


def online_order(ev, corpus, debt, index0, hist_rows, K, change, params, prior_rows,
                 observe_full=False):
    """在线 bandit：返回按修复顺序排列的欠账块列表（长度 ≤ K）。不读任何 gold。

    冷启动用**事故前查询日志**（历史半边在全量索引上的 top-10，同样无 gold）做先验；
    随后每轮用欠账状态下新服务的查询做衰减更新（γ=0.95），体现 restless bandit 的在线估计。
    """
    allrows = np.arange(corpus.n, dtype=np.int64)
    Rds, W = params['rounds'], params['window']
    alpha, c, beta = params['alpha'], params['ucb_c'], params['beta']
    n_pages = corpus.n_pages
    debt_set = np.zeros(corpus.n, dtype=bool)
    debt_set[debt] = True
    Np = np.array([len(ch) for ch in corpus.page_chunks], dtype=np.float64)
    repaired = np.zeros(corpus.n, dtype=bool)
    last_repair = np.zeros(n_pages, dtype=np.int32)
    state = np.sort(index0.copy())
    # 冷启动：事故前日志（全量索引）的排名加权曝光，作为先验；γ 衰减随在线查询推进
    gamma = params.get('gamma', 0.95)
    wsum = np.zeros(n_pages, dtype=np.float64)
    ncnt = np.zeros(n_pages, dtype=np.float64)
    t = 0
    t_eff = 0.0
    for qi in prior_rows:
        t += 1
        t_eff = t_eff * gamma + 1.0
        wsum *= gamma
        ncnt *= gamma
        for rank, p in enumerate(ev.top_pages(np.arange(corpus.n, dtype=np.int64), qi)[:10]):
            wsum[int(p)] += 1.0 / (rank + 1)
            ncnt[int(p)] += 1.0
    chosen: list[int] = []
    groups = [list(g) for g in np.array_split(np.asarray(hist_rows, dtype=np.int64), Rds)]
    quota = [K // Rds] * Rds
    for i in range(K % Rds):
        quota[i] += 1
    for r, group in enumerate(groups):
        for qi in group:
            t += 1
            t_eff = t_eff * gamma + 1.0
            wsum *= gamma
            ncnt *= gamma
            obs_state = allrows if observe_full else state
            for rank, p in enumerate(ev.top_pages(obs_state, qi)[:10]):
                wsum[int(p)] += 1.0 / (rank + 1)
                ncnt[int(p)] += 1.0
        # 在线观测（衰减累计）→ 页优先级
        rate = (wsum + alpha) / (t_eff + 2 * alpha)   # 每查询期望奖励（而非按曝光次数归一）
        ucb = 1.0 + c * np.sqrt(math.log(t + 1.0) / (ncnt + 1.0))
        stale = 1.0 + beta * (r - last_repair) / float(Rds)
        W_online = rate * ucb * stale
        order = L.order_greedy(corpus, W_online, None, chunk_key=change)
        take = quota[r]
        for d in order.tolist():
            if take <= 0:
                break
            if not debt_set[d] or repaired[d]:
                continue
            chosen.append(d)
            repaired[d] = True
            take -= 1
            p = int(corpus.page_ids[d])
            last_repair[p] = r
        # 本轮修复后更新检索状态（只在下一轮生效）
        if chosen:
            state = np.sort(np.concatenate([index0, np.asarray(chosen, dtype=np.int64)]))
    return np.asarray(chosen, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default=str(E3.parent / 'e2' / 'corpus-structural.jsonl.gz'))
    ap.add_argument('--vectors', default=str(E3.parent / 'e2' / 'vectors' / 'vectors.npy'))
    ap.add_argument('--qvecs', default=str(E3.parent / 'zzu-pilot' / 'qvecs.npy'))
    ap.add_argument('--gold', default=str(E3.parent / 'zzu-pilot' / 'gold.json'))
    ap.add_argument('--out', default=str(HERE))
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--budgets', default='300,600,1200')
    ap.add_argument('--scenarios', default='full,hot,edit24')
    ap.add_argument('--seed', type=int, default=20261110)
    ap.add_argument('--alpha', type=float, default=PARAMS['alpha'])
    ap.add_argument('--ucb-c', type=float, default=PARAMS['ucb_c'])
    ap.add_argument('--beta', type=float, default=PARAMS['beta'])
    ap.add_argument('--rounds', type=int, default=PARAMS['rounds'])
    ap.add_argument('--gamma', type=float, default=PARAMS.get('gamma', 0.95))
    ap.add_argument('--observe-full', action='store_true',
                    help='机制消融：在线观测改用全量索引（排除"退化日志"因素）')
    args = ap.parse_args()
    PARAMS.update(rounds=args.rounds, alpha=args.alpha, ucb_c=args.ucb_c,
                  beta=args.beta, gamma=args.gamma)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    budgets = [int(x) for x in args.budgets.split(',')]
    scenarios = [s.strip() for s in args.scenarios.split(',') if s.strip()]

    gold, Q, usable, corpus, ev, rel, ages, change, sec = load_meta(args)
    allrows = np.arange(corpus.n, dtype=np.int64)
    full_q = ev.recall(allrows, usable).astype(np.float64)
    k_by_budget = {b: min(int(b / sec), corpus.n) for b in budgets}
    log(f'corpus={corpus.n} usable={len(usable)} budgets={k_by_budget} params={PARAMS}')

    rows = []
    t0 = time.time()
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
            debt, index0, note = builder(hist_rows, rng)
            before_q = ev.recall(index0, usable).astype(np.float64)
            rb, rf = before_q[eval_pos].mean(), full_q[eval_pos].mean()
            den = rf - rb
            orders, _events = R.strategy_orders(corpus, ev, hist_rows, rel, rng,
                                                allrows, usable, ages, change)
            slot = {'scenario': sc, 'rep': r, 'R_before': float(rb), 'R_full': float(rf),
                    'debt': int(len(debt)), 'note': note}
            for b in budgets:
                K = k_by_budget[b]
                # 非在线臂
                for arm, key in (('greedy-full', 'greedy'), ('greedy-nolabel', 'greedy-nolabel'),
                                 ('fifo', 'fifo'), ('lru', 'lru'), ('random', 'random')):
                    # 与 E3 一致：只在欠账集内按该臂的顺序取前 K 块（不是全语料前 K）
                    rank = np.empty(corpus.n, dtype=np.int64)
                    rank[orders[key]] = np.arange(corpus.n)
                    rest = debt[np.argsort(rank[debt])]
                    state = np.sort(np.concatenate([index0, rest[:K]]))
                    rec = float(ev.recall(state, eval_rows).mean())
                    slot[f'{arm}_recall'] = rec
                    slot[f'{arm}_delta_pp'] = 100 * (rec - rb)
                # 在线 bandit
                chosen = online_order(ev, corpus, debt, index0, hist_rows, K, change, PARAMS,
                                      prior_rows=hist_rows, observe_full=args.observe_full)
                state = np.sort(np.concatenate([index0, chosen]))
                rec = float(ev.recall(state, eval_rows).mean())
                slot['online_recall'] = rec
                slot['online_delta_pp'] = 100 * (rec - rb)
                slot['online_n_embeds'] = int(len(chosen))
                slot['budget_s'] = b
                slot['den'] = float(den)
                rows.append(dict(slot))
            log(f'[{sc} rep{r}] debt={len(debt)} rb={rb:.4f} rf={rf:.4f} '
                + ' '.join(f'{b}s:on={next(x for x in rows if x["scenario"]==sc and x["rep"]==r and x["budget_s"]==b)["online_delta_pp"]:+.1f}'
                           for b in budgets) + 'pp')

    # 汇总
    table = []
    for sc in scenarios:
        for b in budgets:
            for arm in ARMS:
                sub = [x for x in rows if x['scenario'] == sc and x['budget_s'] == b]
                vals = [x[f'{arm}_delta_pp'] for x in sub]
                recs = [x[f'{arm}_recall'] for x in sub]
                # 归一化恢复率：只在每个 rep 的分母>0 且 R_after≤R_full 时给
                dens = [x['R_full'] - x['R_before'] for x in sub]
                den_ok = all(d > 1e-12 for d in dens)
                within = all(x[f'{arm}_recall'] <= x['R_full'] + 1e-12 for x in sub)
                valid = den_ok and within
                recovery = (float(np.mean([(x[f'{arm}_recall'] - x['R_before']) / (x['R_full'] - x['R_before'])
                                           for x in sub])) if valid else None)
                note = ('ok' if valid else
                        ('n/a: R_after > R_full（非单调）' if den_ok else 'n/a: 分母 ≤ 0'))
                table.append({'scenario': sc, 'budget_s': b, 'arm': arm,
                              'delta_pp_mean': float(np.mean(vals)),
                              'delta_pp_median': float(np.median(vals)),
                              'delta_pp_sd': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                              'recall_mean': float(np.mean(recs)),
                              'recovery_mean': recovery, 'recovery_note': note,
                              'per_rep_delta_pp': [round(float(v), 4) for v in vals]})
    # 判据：600s 上 online vs nolabel / full
    def wtest(a, b):
        diffs = [x[f'{a}_delta_pp'] - x[f'{b}_delta_pp']
                 for x in rows if x['scenario'] == 'full' and x['budget_s'] == 600]
        return L.wilcoxon_exact(diffs), diffs
    (wt_nl, d_nl) = wtest('online', 'greedy-nolabel')
    (wt_fl, d_fl) = wtest('online', 'greedy-full')
    gain = {a: float(np.mean([x[f'{a}_delta_pp'] for x in rows
                              if x['scenario'] == 'full' and x['budget_s'] == 600]))
            for a in ARMS}
    ratio = gain['online'] / gain['greedy-full'] if gain['greedy-full'] > 1e-9 else float('nan')
    crit = {
        'scenario': 'full', 'budget_s': 600, 'params': {**PARAMS, 'observe_full': args.observe_full},
        'gain_pp': {k: round(v, 3) for k, v in gain.items()},
        'online_vs_nolabel': {'mean_gap_pp': round(float(np.mean(d_nl)), 3),
                              'wilcoxon_one_sided': wt_nl['p_one_sided'],
                              'wilcoxon_two_sided': wt_nl['p_two_sided'],
                              'rank_biserial': wt_nl['rank_biserial']},
        'online_vs_fulllabel': {'mean_gap_pp': round(float(np.mean(d_fl)), 3),
                                'wilcoxon_one_sided': wt_fl['p_one_sided'],
                                'rank_biserial': wt_fl['rank_biserial']},
        'online_over_full_ratio': round(float(ratio), 3),
    }
    crit['beats_nolabel'] = bool(gain['online'] > gain['greedy-nolabel']
                                 and wt_nl['p_one_sided'] < 0.05)
    crit['reaches_half_of_full'] = bool(ratio >= 0.5)
    crit['practical_value'] = bool(crit['beats_nolabel'] and crit['reaches_half_of_full'])
    log(f"判据：online={gain['online']:+.2f}pp nolabel={gain['greedy-nolabel']:+.2f}pp "
        f"full={gain['greedy-full']:+.2f}pp ratio={ratio:.2f} "
        f"p(online>nolabel)={wt_nl['p_one_sided']:.4f} → practical={crit['practical_value']}")

    with open(out / 'results.json', 'w', encoding='utf-8') as f:
        json.dump({'meta': {'corpus_n': corpus.n, 'reps': args.reps, 'budgets_k': k_by_budget,
                            'scenarios': scenarios, 'params': PARAMS, 'seed': args.seed},
                   'table': table, 'criterion': crit, 'rows': rows}, f, ensure_ascii=False, indent=1)
    with open(out / 'table_detail.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['scenario', 'budget_s', 'arm', 'recall_mean',
                                          'delta_pp_mean', 'delta_pp_median', 'delta_pp_sd',
                                          'recovery_mean', 'recovery_note', 'per_rep_delta_pp'])
        w.writeheader()
        for row in table:
            w.writerow({**row,
                        'recovery_mean': 'n/a' if row['recovery_mean'] is None else row['recovery_mean'],
                        'per_rep_delta_pp': ' '.join(f'{x:.4f}' for x in row['per_rep_delta_pp'])})
    log(f'wrote {out}/table_detail.csv, results.json  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
