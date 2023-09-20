# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: This basic example loads a matching model and use it to
compute cosine similarity for a given list of sentences.
"""
import sys

sys.path.append('..')
from similarities import BertSimilarity

# Two lists of sentences
sentences1 = [
    'The cat sits outside',
    'A man is playing guitar',
    'The new movie is awesome',
    '花呗更改绑定银行卡',
    'The quick brown fox jumps over the lazy dog.',
]

sentences2 = [
    'The dog plays in the garden',
    'A woman watches TV',
    'The new movie is so great',
    '如何更换花呗绑定银行卡',
    '敏捷的棕色狐狸跳过了懒狗',
]

model = BertSimilarity(model_name_or_path="shibing624/text2vec-base-multilingual")
# 使用的是多语言文本匹配模型
scores = model.similarity(sentences1, sentences2)
print('1:use Similarity compute cos scores\n')
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[j], scores[i][j]))

print('-' * 50 + '\n')
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
    'Do you like pizza?',
    '如何更换花呗绑定银行卡',
    '敏捷的棕色狐狸跳过了懒狗',
    '猫在窗外',
    '电影很棒',
]

model.add_corpus(corpus)
model.save_embeddings('en_corpus_emb.json')
res = model.most_similar(queries=sentences1, topn=3)
print(res)
del model
model = BertSimilarity(model_name_or_path="shibing624/text2vec-base-multilingual")
model.load_embeddings('en_corpus_emb.json')
res = model.most_similar(queries=sentences1, topn=3)
print(res)
for q_id, c in res.items():
    print('query:', sentences1[q_id])
    print("search top 3:")
    for corpus_id, s in c.items():
        print(f'\t{model.corpus[corpus_id]}: {s:.4f}')
