# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')

from similarities import (
    BertSimilarity,
    TfidfSimilarity,
    EnsembleSimilarity,
    BM25Similarity,
    CilinSimilarity,
)


def sim_and_search(m):
    print(m)
    if 'BM25' not in str(m):
        sim_scores = m.similarity(text1, text2)
        print('sim scores: ', sim_scores)
        for (idx, i), j in zip(enumerate(text1), text2):
            s = sim_scores[idx] if isinstance(sim_scores, list) else sim_scores[idx][idx]
            print(f"{i} vs {j}, score: {s:.4f}")
    m.add_corpus(corpus)
    res = m.most_similar(queries, topn=3)
    print('sim search: ', res)
    for q_id, c in res.items():
        print('query:', queries[q_id])
        print("search top 3:")
        for corpus_id, s in c.items():
            print(f'\t{m.corpus[corpus_id]}: {s:.4f}')
    print('-' * 50 + '\n')


if __name__ == '__main__':
    text1 = [
        '如何更换花呗绑定银行卡',
        '花呗更改绑定银行卡'
    ]
    text2 = [
        '花呗更改绑定银行卡',
        '我什么时候开通了花呗',
    ]
    corpus = [
        '花呗更改绑定银行卡',
        '我什么时候开通了花呗',
        '俄罗斯警告乌克兰反对欧盟协议',
        '暴风雨掩埋了东北部；新泽西16英寸的降雪',
        '中央情报局局长访问以色列叙利亚会谈',
        '人在巴基斯坦基地的炸弹袭击中丧生',
    ]

    queries = [
        '我的花呗开通了？',
        '乌克兰被俄罗斯警告',
        '更改绑定银行卡',
    ]
    print('text1: ', text1)
    print('text2: ', text2)
    print('query: ', queries)
    print('corpus: ', corpus)
    m1 = BertSimilarity()
    m2 = TfidfSimilarity()
    m3 = CilinSimilarity()
    m4 = BM25Similarity()
    m = EnsembleSimilarity(similarities=[m1, m2, m3, m4], weights=[0.7, 0.1, 0.1, 0.1], c=2)
    sim_and_search(m1)
    sim_and_search(m2)
    sim_and_search(m3)
    sim_and_search(m4)
    sim_and_search(m)
    m.save_corpus_embeddings()
    del m
    m = EnsembleSimilarity(similarities=[m1, m2, m3, m4], weights=[0.7, 0.1, 0.1, 0.1], c=2)
    m.load_corpus_embeddings()
    sim_and_search(m)
