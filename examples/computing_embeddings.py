# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This basic example loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of sentences.
"""

import sys

sys.path.append('..')
from similarities import BertSimilarity

model = BertSimilarity("shibing624/text2vec-base-chinese")  # 中文句向量模型(CoSENT)
# Embed a list of sentences
sentences = ['如何更换花呗绑定银行卡',
             '花呗更改绑定银行卡']
sentence_embeddings = model.encode(sentences)
print(type(sentence_embeddings), sentence_embeddings.shape)
