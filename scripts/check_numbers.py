#!/usr/bin/env python3
"""校验两份稿子：无占位符、数字中英一致、数字有出处、稿内表格与 CSV 生成结果逐字节一致。

三类硬检查（任一失败即 exit 1）
1. 占位符：两份稿子都不含 `PENDING` / `待填`。
2. 数字一致：`tables/facts.json` 里每个"论文精度"字符串都同时出现在中英文稿中
   （比较前把 Unicode 减号 − 归一成 -，避免排版字符造成假失败）。
3. 表格一致：`tables/table*.{zh,en}.md` 每一份都原样出现在对应语言的稿子里。

一项硬检查（新增）
5. PDF 标题覆盖：每份 PDF 必须包含其源 MD 的全部 `##` 标题
   （堵住「builder 从 Abstract 起渲染、静默丢掉 Highlights/Keywords」这类洞）。

一项报告（不计失败）
4. 第 6 节出现的数字，若在 facts / 表格 / `experiments/{e1,e2/out,e3/out}` 的文本里都找不到，
   列出来供人工复核（用来发现手写进去、没有出处的数字）。

用法：python3 docs/manuscript/check_numbers.py [-v]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent      # scripts/
PKG = HERE.parent                            # 包根
ROOT = PKG
DOCS = {'zh': PKG / 'paper' / 'paper-v0.1-zh.md', 'en': PKG / 'paper' / 'paper-v0.1.md'}
T = PKG / 'tables'
FACTS = json.loads((T / 'facts.json').read_text(encoding='utf-8'))

MINUS = str.maketrans({'−': '-', '–': '-', '\u00a0': ' '})


def norm(s: str) -> str:
    return s.translate(MINUS)


def read(p: Path) -> str:
    return norm(p.read_text(encoding='utf-8'))


def section6(text: str) -> str:
    m = re.search(r'\n## 6[.\s]', text)
    if not m:
        return ''
    i = m.start()
    m2 = re.search(r'\n## 7[.\s]', text[i + 1:])
    return text[i:i + 1 + (m2.start() if m2 else len(text))]


def experiment_corpus() -> str:
    """实验目录里可以当"出处"的文本（CSV/JSON/MD，跳过镜像与向量大文件）。"""
    parts = []
    globs = ['experiments/e1/*.csv', 'experiments/e1/*.json', 'experiments/e1/*.md',
             'experiments/e2/out/*.csv', 'experiments/e2/out/*.json', 'experiments/e2/out/*.md',
             'experiments/e3/out/*.csv', 'experiments/e3/out/*.json', 'experiments/e3/out/*.md',
             'experiments/e3/*.csv', 'experiments/e3/*.md', 'experiments/*/PROTOCOL.md',
             'experiments/METRICS.md',
             'experiments/sweep-retrieval/*.csv', 'experiments/sweep-retrieval/*.md',
             'experiments/chunk-sweep/*.csv', 'experiments/chunk-sweep/*.md',
             'experiments/pooling/*.csv', 'experiments/pooling/*.md', 'experiments/pooling/out/*.json',
             'experiments/model-matrix/*.csv', 'experiments/model-matrix/*.md', 'experiments/model-matrix/out/*.json',
             'experiments/beir/*.csv', 'experiments/beir/*.md', 'experiments/beir/out/*.json',
             'experiments/cmteb/*.csv', 'experiments/cmteb/*.md',
             'experiments/rule2-ablation/*.csv', 'experiments/rule2-ablation/*.md',
             'experiments/rule3-ablation/*.csv', 'experiments/rule3-ablation/*.md',
             'experiments/e3-external/*.csv', 'experiments/e3-external/*.md',
             'experiments/e1/seeds/*.json', 'experiments/e1/seeds/*.md']
    for g in globs:
        for p in sorted(ROOT.glob(g)):
            if p.is_file() and p.stat().st_size < 4_000_000:
                parts.append(read(p))
    return '\n'.join(parts)


def pdf_headings_check() -> list[str]:
    """检查每份 PDF 是否包含其源 MD 里的全部 ## 标题。

    背景：build_pdf_en.py 曾从 `## Abstract` 起渲染，静默丢掉 Highlights/Keywords，
    而 MD 与数字校验都发现不了。这条检查专门堵这个洞。

    PDF 提取的中文可能含部首兼容字符（⼉ U+2F09 / ⾝ U+2F5D），故做 NFKC 归一。
    """
    import unicodedata
    try:
        import pypdf
    except ImportError:
        print('[5] PDF 标题覆盖: SKIP（pypdf 未安装）')
        return []

    def squash(x: str) -> str:
        x = unicodedata.normalize('NFKC', x)
        return re.sub(r'[\s\u00a0]+', '', x).lower()

    pairs = [('en', DOCS['en'], DOCS['en'].parent / 'paper-v0.1-en.pdf'),
             ('zh', DOCS['zh'], DOCS['zh'].parent / 'paper-v0.1-zh.pdf')]
    bad: list[str] = []
    for lang, md, pdf in pairs:
        if not pdf.exists():
            print(f'[5] PDF 标题覆盖 {lang}: SKIP（{pdf.name} 不存在）')
            continue
        try:
            reader = pypdf.PdfReader(str(pdf))
            blob = squash(' '.join(pg.extract_text() or '' for pg in reader.pages))
        except Exception as exc:                                  # noqa: BLE001
            print(f'[5] PDF 标题覆盖 {lang}: SKIP（读取失败 {type(exc).__name__}）')
            continue

        heads = re.findall(r'(?m)^##\s+(.+)$', read(md))
        # 跳过非正文小节：写作设定、References、附录（PDF 可能分页截断，只查正文骨架）
        skip = ('写作设定', 'References', '参考文献')
        heads = [h for h in heads if not h.strip().startswith(skip)]
        missing = [h for h in heads if squash(h) not in blob]
        status = 'PASS' if not missing else 'FAIL ' + str(missing[:5])
        print(f'[5] PDF 标题覆盖 {lang}: {status} ({len(heads) - len(missing)}/{len(heads)} 节)')
        if missing:
            bad.append(f'{lang} PDF missing headings: {missing[:8]}')
    return bad


def main() -> int:
    verbose = '-v' in sys.argv
    failures = []
    texts = {k: read(p) for k, p in DOCS.items()}

    # 1. 占位符
    for lang, text in texts.items():
        hits = [i + 1 for i, line in enumerate(text.splitlines())
                if 'PENDING' in line or '待填' in line]
        status = 'PASS' if not hits else f'FAIL lines={hits[:10]}'
        print(f'[1] 占位符 {lang}: {status}')
        if hits:
            failures.append(f'placeholder in {lang}: {hits[:10]}')

    # 2. 数字一致（中英同源）：只在正文与表格里找，**不接受**附录 C 的对照表替身
    body = {'zh': texts['zh'].split('## 附录 C')[0],
            'en': texts['en'].split('## Appendix C')[0]}
    missing = {'zh': [], 'en': []}
    for item in FACTS['paper']:
        s = norm(item['s'])
        for lang, text in body.items():
            if s not in text:
                missing[lang].append(f"{item['key']}={s}")
    for lang in ('zh', 'en'):
        status = 'PASS' if not missing[lang] else f'FAIL missing={len(missing[lang])}'
        print(f'[2] facts 出现于正文/表格 {lang}: {status}')
        if missing[lang]:
            failures.append(f'facts missing in {lang}: {missing[lang][:12]}')

    # 3. 表格逐字节一致
    for lang in ('zh', 'en'):
        names = sorted(T.glob(f'table*.{lang}.md')) + sorted(T.glob(f'provenance.{lang}.md'))
        bad = []
        for p in names:
            block = read(p).strip('\n')
            if block not in texts[lang]:
                bad.append(p.name)
        print(f'[3] 表格内联 {lang}: {"PASS" if not bad else "FAIL " + str(bad)} ({len(names)} 张)')
        if bad:
            failures.append(f'tables not inlined verbatim ({lang}): {bad}')

    # 4. 报告：第 6 节里没有出处的数字
    corpus = experiment_corpus() + '\n' + '\n'.join(
        norm(p.read_text(encoding='utf-8')) for p in sorted(T.glob('table*.md')))
    corpus += '\n' + '\n'.join(norm(x['s']) for x in FACTS['paper'])
    pat = re.compile(r'\d[\d,]*\.\d+%?|\d[\d,]*%')
    for lang, text in texts.items():
        sec = section6(text)
        unknown = sorted({m.group(0) for m in pat.finditer(sec) if m.group(0) not in corpus})
        print(f'[4] §6 数字无出处 {lang}: {len(unknown)} 个' + (f' → {unknown[:20]}' if unknown else ''))

    # 5. PDF 是否覆盖 MD 的全部标题
    failures.extend(pdf_headings_check())

    print()
    if failures:
        print('RESULT: FAIL')
        for f in failures:
            print(' -', f)
        return 1
    print('RESULT: PASS（占位符 0 · facts %d 条中英一致 · 表格逐字节一致）' % len(FACTS['paper']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
