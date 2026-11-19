"""生成独立的新查询集：从语料里挑页面，让 LLM 写一个答案在文中的问题。

为什么：论文的 109 题只对应 48 个 gold page（reuse 2.68），
query 级划分会让 55.7% 的历史侧 gold 页出现在评测侧。要验证 §5.3b
的结论不是这一套题的特例，需要一批**全新、与旧题不重叠**的查询。

用法：python3 gen.py --n 1000 [--dry 5]
"""
import argparse, json, os, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / 'hpc' / 'pylibs'))
import e2_lib as L

ENV = {}
for ln in open('/data/agent/.dsh/secrets/mimo.env'):
    ln = ln.strip()
    if '=' in ln and not ln.startswith('#'):
        k, v = ln.split('=', 1)
        ENV[k.strip()] = v.strip().strip('"').strip("'")
KEY = ENV['DEEPSEEK_API_KEY']
URL = 'https://api.deepseek.com/v1/chat/completions'
MODEL = 'deepseek-chat'

PROMPT = '''下面是一段技术知识库的正文。请**只**根据这段文字写一个具体问题，
要求：
1. 问题的答案必须能在文中直接找到（不要问需要外部知识的问题）；
2. 问题要具体，包含文中的关键实体或术语；
3. 只输出问题本身，不要加引号、编号或任何解释；
4. 用中文提问，一句话，不超过 40 字。

正文：
{text}'''

def call(text):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
        "max_tokens": 80, "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        'Content-Type': 'application/json', 'Authorization': f'Bearer {KEY}'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            q = d['choices'][0]['message']['content'].strip()
            q = q.strip('"').strip("'").strip()
            if len(q) >= 6:
                return q, d.get('usage', {})
        except Exception as e:
            if attempt == 2:
                return None, str(e)[:60]
            time.sleep(2 * (attempt + 1))
    return None, 'empty'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20261119)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--dry', type=int, default=0)
    a = ap.parse_args()

    corpus = L.load_corpus(str(HERE.parent / 'corpus-structural.jsonl.gz'))
    # 现有 gold 页
    gold = json.load(open(HERE.parent.parent / 'zzu-pilot' / 'gold.json'))
    used = set()
    for qid, spans in gold['qrels'].items():
        for sp in spans:
            used.add(sp['path'])
    cand = [p for p in corpus.pages if p not in used]
    print(f'页面 {len(corpus.pages)}，旧题占用 {len(used)}，可选 {len(cand)}')

    by_page = {}
    for i, p in enumerate(corpus.paths):
        by_page.setdefault(p, []).append(i)

    rng = random.Random(a.seed)
    take = rng.sample(cand, min(a.n, len(cand)))
    if a.dry:
        take = take[:a.dry]

    tasks = []
    for p in take:
        idxs = by_page[p]
        # 取该页最长的一块做种子，长度控制在 300-900 字
        best = max(idxs, key=lambda i: len(corpus.texts[i]))
        t = corpus.texts[best]
        if len(t) > 900:
            t = t[:900]
        if len(t) < 120:
            continue
        tasks.append((p, t))
    print(f'待生成 {len(tasks)} 题，并发 {a.workers}')

    out = HERE / ('dry.jsonl' if a.dry else 'queries-1000.jsonl')
    done = 0
    tok_in = tok_out = 0
    with out.open('w', encoding='utf-8') as f, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(call, t): p for p, t in tasks}
        for fu in as_completed(futs):
            p = futs[fu]
            q, usage = fu.result()
            done += 1
            if q:
                if isinstance(usage, dict):
                    tok_in += usage.get('prompt_tokens', 0)
                    tok_out += usage.get('completion_tokens', 0)
                f.write(json.dumps({'qid': f'iq-{done:04d}', 'question': q,
                                    'gold_page': p}, ensure_ascii=False) + '\n')
            if done % 50 == 0:
                print(f'  {done}/{len(tasks)}  已写 {sum(1 for _ in out.open()) if False else ""}', flush=True)
    n_ok = sum(1 for _ in out.open())
    print(f'完成：{n_ok} 题写入 {out}')
    print(f'用量：输入 {tok_in:,} / 输出 {tok_out:,} tokens ≈ {(tok_in*1e-6*1 + tok_out*1e-6*2):.2f} 元')

if __name__ == '__main__':
    raise SystemExit(main())
