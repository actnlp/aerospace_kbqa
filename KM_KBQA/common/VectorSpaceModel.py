from functools import lru_cache

import torch
import torch.nn.functional as F

from ..common.HITBert import encode, encode_batch


class VectorSpaceModel():
    def __init__(self, docs):
        self.docs = docs
        self.vectors = encode_batch(docs)

    def _most_similar(self, doc, sim_func, topn=3):
        v = encode(doc)
        scores = sim_func(v[None, :], self.vectors)
        sort_idx = torch.argsort(scores, descending=True)
        res = [self.docs[idx] for idx in sort_idx[:topn]]
        return res

    @lru_cache(maxsize=4096)
    def most_similar_cosine(self, doc, topn=3):
        return self._most_similar(doc, F.cosine_similarity, topn=topn)

    @lru_cache(maxsize=4096)
    def most_similar_l2(self, doc, topn=3):
        return self._most_similar(doc, F.pairwise_distance, topn=topn)


if __name__ == "__main__":
    docs = ['男人', '女人', '国王', '皇后']
    w = '美女'
    vsm = VectorSpaceModel(docs)
    print(vsm.most_similar_cosine(w))
