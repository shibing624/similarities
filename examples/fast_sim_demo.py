# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Fast similarity search demo
"""
import os
import sys

sys.path.append('..')
from similarities.fastsim import AnnoySimilarity
from similarities.fastsim import HnswlibSimilarity

sentences = ['如何更换花呗绑定银行卡',
             '花呗更改绑定银行卡']
corpus = [
    '花呗更改绑定银行卡',
    '我什么时候开通了花呗',
    '俄罗斯警告乌克兰反对欧盟协议',
    '暴风雨掩埋了东北部；新泽西16英寸的降雪',
    '中央情报局局长访问以色列叙利亚会谈',
    '人在巴基斯坦基地的炸弹袭击中丧生',
    '我喜欢这首歌'
]


def hnswlib_demo():
    model = HnswlibSimilarity(corpus=corpus * 10)
    print(model)
    similarity_score = model.similarity(sentences[0], sentences[1])
    print(f"{sentences[0]} vs {sentences[1]}, score: {float(similarity_score):.4f}")
    model.add_corpus(corpus)
    model.build_index()
    model.save_index('test.model')
    # Semantic Search batch
    print(model.most_similar("men喜欢这首歌"))
    queries = ["如何更换花呗绑定银行卡", "men喜欢这首歌"]
    res = model.most_similar(queries, topn=3)
    for q, r in zip(queries, res):
        print(f"{q} -> {r}")
    os.remove('test.model')
    print('-' * 50 + '\n')

def annoy_demo():
    model = AnnoySimilarity(corpus=corpus * 10)
    print(model)
    similarity_score = model.similarity(sentences[0], sentences[1])
    print(f"{sentences[0]} vs {sentences[1]}, score: {float(similarity_score):.4f}")
    model.add_corpus(corpus)
    model.build_index()
    model.save_index('test.model')
    # Semantic Search batch
    print(model.most_similar("men喜欢这首歌"))
    queries = ["如何更换花呗绑定银行卡", "men喜欢这首歌"]
    res = model.most_similar(queries, topn=3)
    for q, r in zip(queries, res):
        print(f"{q} -> {r}")
    os.remove('test.model')
    print('-' * 50 + '\n')

if __name__ == '__main__':
    # hnswlib_demo()
    annoy_demo()
