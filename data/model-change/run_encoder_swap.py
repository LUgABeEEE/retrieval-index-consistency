"""model change 的正确对照：旧 store 的行在新模型下全部失去身份。

naive：照旧把旧行喂进排序 → PER 应该高
protocol：按 identity 过滤 → 旧行被排除，recall 崩但 PER=0
"""
import json, sys
from pathlib import Path
HERE = Path('/data/agent/workspace/dsh-kb/experiments/e2')
MM = HERE.parent / 'model-matrix' / 'vectors'
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'hpc' / 'pylibs'))
import numpy as np
import e2_lib as L

corpus = L.load_corpus(str(HERE / 'corpus-structural.jsonl.gz'))
gold = json.load(open(HERE.parent / 'zzu-pilot' / 'gold.json'))
qs = [q for q in gold['questions'] if q['qid'] in gold['qrels']]
qrels, spans = gold['qrels'], L.page_spans(corpus)
bm25 = L.BM25(corpus.texts)
pos = {q['qid']: i for i, q in enumerate(gold['questions'])}

def qv(model):
    q = np.load(MM / model / 'q_cls.npy').astype(np.float32)
    return np.stack([q[pos[x['qid']]] for x in qs])

V_old = np.load(MM / 'bge-m3/fp16-cls/corpus_cls.npy').astype(np.float32)
qv_new = qv('bge-large-zh')


class Store2(L.Store):
    """stale=True 时：行保留但 chunk=None（naive 行为）；
       stale=False 时：行被过滤掉（protocol 行为）。"""
    def __init__(self, V, stale):
        self.V, self.stale = V, stale
    def load(self, c):
        if self.stale:
            rmeta = [L.RowMeta(c.paths[i], c.starts[i], c.texts[i], None) for i in range(len(c.paths))]
            return self.V, rmeta
        # protocol：identity 不匹配的行不加载
        return np.zeros((0, self.V.shape[1]), np.float32), []

for name, stale in (('naive (old rows kept)', True), ('protocol (identity-filtered)', False)):
    ev = L.evaluate_serial(corpus, bm25, Store2(V_old, stale), qs, qv_new, qrels, spans)
    print(f'  {name:<30} recall@5={ev["recall5"]:.4f}  PER@10={ev["per10"]*100:6.2f}%  陈行={ev["n_stale_rows"]}')

# 参照：新模型自己重建
V_new = np.load(MM / 'bge-large-zh/corpus_cls.npy').astype(np.float32)
ev = L.evaluate_serial(corpus, bm25, L.FixedStore(V_new), qs, qv_new, qrels, spans) if hasattr(L, 'FixedStore') else None
if ev is None:
    class FS(L.Store):
        def __init__(self, V): self.V = V
        def load(self, c):
            return self.V, [L.RowMeta(c.paths[i], c.starts[i], c.texts[i], i) for i in range(len(c.paths))]
    ev = L.evaluate_serial(corpus, bm25, FS(V_new), qs, qv_new, qrels, spans)
print(f'  {"full re-embed, new model":<30} recall@5={ev["recall5"]:.4f}  PER@10={ev["per10"]*100:6.2f}%  陈行={ev["n_stale_rows"]}')
