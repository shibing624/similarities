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


def main():
    text1 = '刘若英是个演员'
    text2 = '他唱歌很好听'
    m = SimHashSimilarity()
    print(m.similarity(text1, text2))
    print(m.distance(text1, text2))
    print(m.most_similar('刘若英是演员'))
    zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
    m.add_corpus(zh_list)
    print(m.most_similar('刘若英是演员'))

    text1 = "如何更换花呗绑定银行卡"
    text2 = "花呗更改绑定银行卡"
    m = TfidfSimilarity()
    print(text1, text2, ' sim score: ', m.similarity(text1, text2))
    print('distance:', m.distance(text1, text2))
    zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌', '我不是演员吗']
    m.add_corpus(zh_list)
    print(m.most_similar('刘若英是演员'))

    m = BM25Similarity()
    zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌', '我不是演员吗']
    m.add_corpus(zh_list)
    print(m.most_similar('刘若英是演员'))

    wm = Word2Vec()
    list_of_corpus = ["This is a test1", "This is a test2", "This is a test3"]
    list_of_corpus2 = ["that is test4", "that is a test5", "that is a test6"]
    m = WordEmbeddingSimilarity(wm, list_of_corpus)
    m.add_corpus(list_of_corpus2)
    v = m._get_vector("This is a test1")
    print(v[:10], v.shape)
    print(m.similarity("This is a test1", "that is a test5"))
    print(m.distance("This is a test1", "that is a test5"))
    print(m.most_similar("This is a test1"))

    text1 = '周杰伦是一个歌手'
    text2 = '刘若英是个演员'
    m = CilinSimilarity()
    print(m.similarity(text1, text2))
    print(m.distance(text1, text2))
    zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
    m.add_corpus(zh_list)
    print(m.most_similar('刘若英是演员'))

    m = HownetSimilarity()
    print(m.similarity(text1, text2))
    print(m.distance(text1, text2))
    zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
    m.add_corpus(zh_list)
    print(m.most_similar('刘若英是演员'))


if __name__ == '__main__':
    main()
