# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
from text2vec import Word2Vec
from loguru import logger

sys.path.append('..')

from similarities.literalsim import SimHashSimilarity, TfidfSimilarity, BM25Similarity, WordEmbeddingSimilarity, \
    CilinSimilarity, HownetSimilarity

logger.remove()
logger.add(sys.stderr, level="INFO")


def simhash_demo():
    m = SimHashSimilarity()
    print(m)
    sim_scores = m.similarity(text1, text2)
    print('sim scores: ', sim_scores)
    for i, j, s in zip(text1, text2, sim_scores):
        print(f"{i} vs {j}, score: {s:.4f}")
    m.add_corpus(corpus)
    search_res = m.most_similar(queries, topn=3)
    print('sim search: ', search_res)
    for q, r in zip(queries, search_res):
        print(f"query: {q}\t search result: {r}")
    print('-' * 50 + '\n')


def tfidf_demo():
    m = TfidfSimilarity()
    print(m)
    sim_scores = m.similarity(text1, text2).numpy()
    print('sim scores: ', sim_scores)
    for (idx, i), j in zip(enumerate(text1), text2):
        s = sim_scores[idx][idx]
        print(f"{i} vs {j}, score: {s:.4f}")
    m.add_corpus(corpus)
    search_res = m.most_similar(queries, topn=3)
    print('sim search: ', search_res)
    for q, r in zip(queries, search_res):
        print(f"query: {q}\t search result: {r}")
    print('-' * 50 + '\n')


def bm25_demo():
    m = BM25Similarity()
    print(m)
    m.add_corpus(corpus)
    search_res = m.most_similar(queries, topn=3)
    print('sim search: ', search_res)
    for q, r in zip(queries, search_res):
        print(f"query: {q}\t search result: {r}")
    print('-' * 50 + '\n')


def wordemb_demo():
    wm = Word2Vec()
    m = WordEmbeddingSimilarity(wm)
    print(m)
    sim_scores = m.similarity(text1, text2)
    print('sim scores: ', sim_scores)
    for (idx, i), j in zip(enumerate(text1), text2):
        s = sim_scores[idx][idx]
        print(f"{i} vs {j}, score: {s:.4f}")
    m.add_corpus(corpus)
    search_res = m.most_similar(queries, topn=3)
    print('sim search: ', search_res)
    for q, r in zip(queries, search_res):
        print(f"query: {q}\t search result: {r}")
    print('-' * 50 + '\n')


def cilin_demo():
    m = CilinSimilarity()
    print(m)
    sim_scores = m.similarity(text1, text2)
    print('sim scores: ', sim_scores)
    for i, j, s in zip(text1, text2, sim_scores):
        print(f"{i} vs {j}, score: {s:.4f}")
    m.add_corpus(corpus)
    search_res = m.most_similar(queries, topn=3)
    print('sim search: ', search_res)
    for q, r in zip(queries, search_res):
        print(f"query: {q}\t search result: {r}")
    print('-' * 50 + '\n')


def hownet_demo():
    m = HownetSimilarity()
    print(m)
    sim_scores = m.similarity(text1, text2)
    print('sim scores: ', sim_scores)
    for i, j, s in zip(text1, text2, sim_scores):
        print(f"{i} vs {j}, score: {s:.4f}")
    m.add_corpus(corpus)
    search_res = m.most_similar(queries, topn=3)
    print('sim search: ', search_res)
    for q, r in zip(queries, search_res):
        print(f"query: {q}\t search result: {r}")


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
        '乌克兰被俄罗斯警告'
    ]
    print('text1: ', text1)
    print('text2: ', text2)
    print('query: ', queries)
    simhash_demo()
    tfidf_demo()
    bm25_demo()
    wordemb_demo()
    cilin_demo()
    hownet_demo()
