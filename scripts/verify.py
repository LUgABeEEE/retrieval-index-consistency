#!/usr/bin/env python3
"""数字溯源抽样器（包内版）：从论文里随机抽数字，去 data/ 里找出处。

用法:
  python3 scripts/verify.py --sample 20     # 抽 20 个
  python3 scripts/verify.py --num 19.72     # 查指定数字
  python3 scripts/verify.py --list          # 只列出抽到的数字

设计取舍：
  * 只做「字面匹配」——在 CSV/JSON/MD 里找同一个数字字符串。
    找不到**不代表错**：可能是算出来的（百分比、均值、差值），也可能是格式不同。
    脚本把「找不到」单列，交给人工判断。
"""
import argparse
import pathlib
import random
import re
import sys

# Windows GBK 控制台对 ✓/✗/◐ 会 UnicodeEncodeError；尽量切到 UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = pathlib.Path(__file__).resolve().parent.parent   # 包根
DATA = HERE / 'data'
CAND = [HERE / 'paper' / 'paper-v0.1-zh.md', HERE / 'paper' / 'paper-v0.1.md']
PAPER = next((c for c in CAND if c.exists()), CAND[-1])

NUM_RE = re.compile(r'\b\d+\.\d+\b|\b\d{1,3}(?:,\d{3})+\b')


def load_numbers():
    text = PAPER.read_text(encoding='utf-8')
    text = re.sub(r'^\|[-:\s|]+\|$', '', text, flags=re.M)   # 去掉表分隔行
    text = re.sub(r'```.*?```', '', text, flags=re.S)        # 去掉代码块
    seen, out = set(), []
    for m in NUM_RE.finditer(text):
        v = m.group(0)
        if v in seen:
            continue
        if re.fullmatch(r'\d{4}\.\d{4,5}', v):               # 跳过 arXiv ID
            continue
        seen.add(v)
        a, b = max(0, m.start() - 40), min(len(text), m.end() + 40)
        out.append((v, re.sub(r'\s+', ' ', text[a:b]).strip()))
    return out


def index_files():
    return sorted(p for p in DATA.rglob('*') if p.is_file() and p.suffix in {'.csv', '.json', '.md'})


def grep_number(files, num, limit=3):
    """返回 (总命中数, 前 limit 处)。"""
    hits, total = [], 0
    for p in files:
        try:
            with open(p, encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f, 1):
                    if num in line:
                        total += 1
                        if len(hits) < limit:
                            hits.append((str(p.relative_to(HERE)), i, line.strip()[:110]))
        except OSError:
            continue
    return total, hits


def variants_of(v, ctx):
    """生成格式变体：百分比 ↔ 小数、去千分位。"""
    out = [v]
    if ',' in v:
        out.append(v.replace(',', ''))
    try:
        f = float(v.replace(',', ''))
        cand = f / 100.0 if ('%' in ctx or f > 1) else f * 100.0
        for nd in (4, 5, 6):
            alt = f'{cand:.{nd}f}'.rstrip('0').rstrip('.')
            if alt and alt != v and len(alt) >= 4:
                out.append(alt)
                break
    except ValueError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=20)
    ap.add_argument('--num', type=str)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--seed', type=int, default=20261114)
    ap.add_argument('--paper', type=str, default=None,
                    help='论文源文件路径（默认在包内 paper/ 下找）')
    a = ap.parse_args()

    global PAPER
    if a.paper:
        PAPER = pathlib.Path(a.paper)
    if not PAPER.exists():
        sys.exit(f'找不到论文：{PAPER}')

    nums = load_numbers()
    print(f'论文：{PAPER.name}')
    print(f'共 {len(nums)} 个不同的数字（至少两位有效数字）\n')

    if a.list:
        for v, ctx in nums[:80]:
            print(f'  {v:>12}   …{ctx}…')
        return

    targets = ([(a.num, '（手工指定）')] if a.num
               else random.Random(a.seed).sample(nums, min(a.sample, len(nums))))

    files = index_files()
    print(f'搜索范围：{len(files)} 个文件（data/ 下 CSV/JSON/MD）\n')

    found, many, missing = [], [], []
    for v, ctx in targets:
        total, hits = 0, []
        for vv in variants_of(v, ctx):
            t, h = grep_number(files, vv)
            if t:
                total, hits = t, [(f'{vv} @ {x[0]}', x[1], x[2]) for x in h]
                break
        if total == 1:
            found.append((v, hits[0], ctx))
            print(f'  ✓ {v:>12}  {hits[0][0]}:{hits[0][1]}')
        elif total > 1:
            many.append((v, total, hits, ctx))
            print(f'  ◐ {v:>12}  {total} 处（如 {hits[0][0]}:{hits[0][1]}）')
        else:
            missing.append((v, ctx))
            print(f'  ✗ {v:>12}  未找到 ← 需人工确认')

    print(f'\n=== 汇总 ===')
    print(f'  唯一命中 {len(found)} · 多处命中 {len(many)} · 未找到 {len(missing)} / 共 {len(targets)}')

    if missing:
        print(f'\n=== 未找到的 {len(missing)} 个 ===')
        for v, ctx in missing:
            print(f'  {v:>12}   …{ctx}…')
        print('\n注意：未找到不代表错误。可能是：')
        print('  · 计算值（均值/百分比/差值），CSV 里没有原样字符串')
        print('  · 格式不同（0.5046 vs 50.46% vs 0.504587）')
        print('  · 见 校准报告.html 第七节（计算值未登记清单）')


if __name__ == '__main__':
    main()
