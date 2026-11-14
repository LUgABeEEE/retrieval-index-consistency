"""BM25 向量化检索：把 postings 预转 numpy，每题按 term 批量累加（bincount），
替代 e2_lib.BM25.search 的逐 posting Python 循环。数值应与原实现一致。"""
import numpy as np
import sys
sys.path.insert(0, r'E:\AI\jobs\recompute')
import e2_lib as L


class BM25Vec:
    def __init__(self, texts):
        self.base = L.BM25(texts)
        b = self.base
        self.n = b.N
        self.k1, self.b, self.avg = b.K1, b.B, b.avgdl
        self.doclen = np.asarray(b.doclen, dtype=np.float32)
        self.idf = b.idf
        # array('i') 交错 [doc, tf, ...] → (m,2) int32
        self.post = {t: np.frombuffer(a, dtype=np.int32).reshape(-1, 2)
                     for t, a in b.post.items() if len(a)}
        self.tokens = L.tokens

    def scores(self, query):
        """返回长度 n 的 float32 分数向量。"""
        sc = np.zeros(self.n, dtype=np.float32)
        k1, bb, avg, dl_all = self.k1, self.b, self.avg, self.doclen
        for t in self.tokens(query):
            a = self.post.get(t)
            if a is None:
                continue
            w = self.idf[t]
            docs = a[:, 0]
            tfs = a[:, 1].astype(np.float32)
            dl = dl_all[docs]
            contrib = w * tfs * (k1 + 1.0) / (tfs + k1 * (1.0 - bb + bb * dl / avg))
            sc += np.bincount(docs, weights=contrib, minlength=self.n).astype(np.float32)
        return sc

    def topk(self, query, depth=50):
        sc = self.scores(query)
        if depth >= self.n:
            idx = np.argsort(-sc)
        else:
            idx = np.argpartition(-sc, depth - 1)[:depth]
            idx = idx[np.argsort(-sc[idx])]
        return [(int(i), float(sc[i])) for i in idx if sc[i] > 0]


if __name__ == '__main__':
    import json, time, collections
    B = r'E:\AI\jobs\recompute'
    ch = [json.loads(l) for l in open(B + r'\chunks-arguana-fixed.jsonl', encoding='utf-8') if l.strip()]
    texts = [c['text'] for c in ch]
    print(f'chunks {len(texts):,}', flush=True)

    t0 = time.time(); vec = BM25Vec(texts); print(f'BM25Vec build {time.time()-t0:.1f}s', flush=True)
    t0 = time.time(); old = L.BM25(texts); print(f'BM25(old) build {time.time()-t0:.1f}s', flush=True)

    qs = [json.loads(l)['text'] for l in open(B + r'\queries-arguana.jsonl', encoding='utf-8')][:50]
    # 数值一致性
    mism = 0
    for q in qs[:10]:
        a = dict(vec.topk(q, 50)); b = dict(old.search(q, 50))
        common = set(a) & set(b)
        if len(common) < 45: mism += 1
        else:
            d = max(abs(a[k] - b[k]) for k in common)
            if d > 1e-3: mism += 1
    print(f'一致性检查: 10 题中 {mism} 题不符', flush=True)

    t0 = time.time()
    for q in qs: vec.topk(q, 50)
    tv = (time.time() - t0) / len(qs)
    t0 = time.time()
    for q in qs: old.search(q, 50)
    to = (time.time() - t0) / len(qs)
    print(f'单题: 向量化 {tv*1000:.1f}ms  vs 原版 {to*1000:.1f}ms  = {to/tv:.1f}x', flush=True)
    print('BM25VEC_DONE', flush=True)
