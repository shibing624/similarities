# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This basic example loads a pre-trained model from the web and uses it to
compute cosine similarity for a given list of sentences.
"""
import sys

sys.path.append('..')
from similarities import Similarity
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

if __name__ == '__main__':
    model = Similarity("shibing624/text2vec-base-chinese")
    # 1.Compute cosine similarity between two sentences.
    sentences = ['如何更换花呗绑定银行卡',
                 '花呗更改绑定银行卡']
    corpus = [
        '花呗更改绑定银行卡',
        '我什么时候开通了花呗',
        '俄罗斯警告乌克兰反对欧盟协议',
        '暴风雨掩埋了东北部；新泽西16英寸的降雪',
        '中央情报局局长访问以色列叙利亚会谈',
        '人在巴基斯坦基地的炸弹袭击中丧生',
    ]
    similarity_score = model.similarity(sentences[0], sentences[1])
    print(f"{sentences[0]} vs {sentences[1]}, score: {float(similarity_score):.4f}")

    # 2.Compute similarity between two list
    similarity_scores = model.similarity(sentences, corpus)
    print(similarity_scores.numpy())
    for i in range(len(sentences)):
        for j in range(len(corpus)):
            print(f"{sentences[i]} vs {corpus[j]}, score: {similarity_scores.numpy()[i][j]:.4f}")

    # 3.Semantic Search
    m = Similarity(sentence_model="shibing624/text2vec-base-chinese", corpus=corpus)
    q = '如何更换花呗绑定银行卡'
    print(m.most_similar(q, topn=5))
    print("query:", q)
    for i in m.most_similar(q, topn=5):
        print('\t', i)
