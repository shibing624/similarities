# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys

sys.path.append('..')
from text2vec import SentenceModel
from similarities.fastsim import AnnoySimilarity
from similarities.fastsim import HnswlibSimilarity

sm = SentenceModel()


def hnswlib():
    list_of_docs = ["This is a test1", "This is a test2", "This is a test3", '刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
    list_of_docs2 = ["that is test4", "that is a test5", "that is a test6", '刘若英个演员', '唱歌很好听', 'men喜欢这首歌']

    m = HnswlibSimilarity(sm, embedding_size=384, corpus=list_of_docs * 10)
    print(m)
    v = m._get_vector("This is test1")
    print(v[:10], v.shape)
    print(m.similarity("This is a test1", "that is a test5"))
    print(m.distance("This is a test1", "that is a test5"))
    print(m.most_similar("This is a test4"))
    print(m.most_similar("men喜欢这首歌"))
    m.add_corpus(list_of_docs2)
    m.build_index()
    print(m.most_similar("This is a test4"))
    print(m.most_similar("men喜欢这首歌"))

    m.save_index('test.model')
    m.load_index('test.model')
    print(m.most_similar("This is a test4"))
    print(m.most_similar("men喜欢这首歌"))
    os.remove('test.model')


def annoy():
    list_of_docs = ["This is a test1", "This is a test2", "This is a test3", '刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
    list_of_docs2 = ["that is test4", "that is a test5", "that is a test6", '刘若英个演员', '唱歌很好听', 'men喜欢这首歌']

    m = AnnoySimilarity(sm, embedding_size=384, corpus=list_of_docs * 10)
    print(m)
    v = m._get_vector("This is test1")
    print(v[:10], v.shape)
    print(m.similarity("This is a test1", "that is a test5"))
    print(m.distance("This is a test1", "that is a test5"))
    print(m.most_similar("This is a test4"))
    print(m.most_similar("men喜欢这首歌"))
    m.add_corpus(list_of_docs2)
    m.build_index()
    print(m.most_similar("This is a test4"))
    print(m.most_similar("men喜欢这首歌"))

    m.save_index('test.model')
    m.load_index('test.model')
    print(m.most_similar("This is a test4"))
    print(m.most_similar("men喜欢这首歌"))
    os.remove('test.model')


if __name__ == '__main__':
    hnswlib()
    annoy()
