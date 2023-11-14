# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from similarities import BertSimilarity

m = BertSimilarity()
sentences1 = [
    '如何更换花呗绑定银行卡',
    '花呗更改绑定银行卡',
]
sentences2 = [
    '花呗更改绑定银行卡',
    '我什么时候开通了花呗',
]

similarity_scores = m.similarity(sentences1, sentences2)
print(f"similarity score: {similarity_scores.numpy()}")
if len(sentences1) == len(sentences2):
    for i in range(len(sentences1)):
        print(f"{sentences1[i]} vs {sentences2[i]}, score: {similarity_scores.numpy()[i][i]:.4f}")
else:
    for i in range(len(sentences1)):
        for j in range(len(sentences2)):
            print(f"{sentences1[i]} vs {sentences2[j]}, score: {similarity_scores.numpy()[i][j]:.4f}")
