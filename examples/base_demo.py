# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This basic example loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of sentences.
"""
import sys

sys.path.append('..')
from similarities.literalsim import WordEmbeddingSimilarity
from text2vec import Word2Vec
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

if __name__ == '__main__':
    wv_model = Word2Vec()
    model = WordEmbeddingSimilarity(wv_model)
    # Embed a list of sentences
    sentences = ['如何更换花呗绑定银行卡',
                 '花呗更改绑定银行卡']
    sentences2 = ['如何更换 银行卡',
                  '西方开花北方结果']
    sentence_embeddings = model.get_vector(sentences)
    print(type(sentence_embeddings), sentence_embeddings.shape)
    similarity_score = model.similarity(sentences[0], sentences[1])
    print(similarity_score.numpy())

    similarity_score = model.similarity(sentences, sentences2)
    print(similarity_score.numpy())
