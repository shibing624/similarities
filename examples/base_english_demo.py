# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: This basic example loads a matching model and use it to
compute cosine similarity for a given list of sentences.
"""
import sys

sys.path.append('..')
from similarities import Similarity

# Two lists of sentences
sentences1 = ['The cat sits outside',
              'A man is playing guitar',
              'The new movie is awesome']

sentences2 = ['The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']

m = Similarity("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# 使用的是多语言文本匹配模型
scores = m.similarity(sentences1, sentences2)
print('1:use Similarity compute cos scores\n')
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[j], scores[i][j]))
print()

print('-' * 42)
print('2:search\n')
# 2.Semantic Search
corpus = [
    'The cat sits outside',
    'A man is playing guitar',
    'I love pasta',
    'The new movie is awesome',
    'The cat plays in the garden',
    'A woman watches TV',
    'The new movie is so great',
    'Do you like pizza?'
]

m.add_corpus(corpus=corpus)
q = 'The cat sits outside'
print(m.most_similar(q, topn=5))
print("query:", q)
for i in m.most_similar(q, topn=5):
    print('\t', i)
