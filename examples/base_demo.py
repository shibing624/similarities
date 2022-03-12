# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 文本语义相似度计算和文本匹配搜索
"""
import sys

sys.path.append('..')
from similarities import Similarity

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
model = Similarity(model_name_or_path="shibing624/text2vec-base-chinese")
print(model)
similarity_score = model.similarity(sentences[0], sentences[1])
print(f"{sentences[0]} vs {sentences[1]}, score: {float(similarity_score):.4f}")

print('-' * 50 + '\n')
# 2.Compute similarity between two list
similarity_scores = model.similarity(sentences, corpus)
print(similarity_scores.numpy())
for i in range(len(sentences)):
    for j in range(len(corpus)):
        print(f"{sentences[i]} vs {corpus[j]}, score: {similarity_scores.numpy()[i][j]:.4f}")

print('-' * 50 + '\n')
# 3.Semantic Search
model.add_corpus(corpus)
res = model.most_similar(queries=sentences, topn=3)
print(res)
for q_id, c in res.items():
    print('query:', sentences[q_id])
    print("search top 3:")
    for corpus_id, s in c.items():
        print(f'\t{model.corpus[corpus_id]}: {s:.4f}')
