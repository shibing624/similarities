# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys
import unittest

sys.path.append('..')
from similarities.fastsim import AnnoySimilarity
from similarities.fastsim import HnswlibSimilarity


class FastTestCase(unittest.TestCase):

    def test_sim_diff(self):
        a = '研究团队面向国家重大战略需求追踪国际前沿发展借鉴国际人工智能研究领域的科研模式有效整合创新资源解决复'
        b = '英汉互译比较语言学'
        m = HnswlibSimilarity()
        r = float(m.similarity(a, b)[0])
        print(a, b, r)
        self.assertTrue(abs(r - 0.4098) < 0.001)
        m = AnnoySimilarity()
        r = float(m.similarity(a, b)[0])
        print(a, b, r)
        self.assertTrue(abs(r - 0.4098) < 0.001)


    def test_hnsw_score(self):
        list_of_docs = ["This is a test1", "This is a test2", "This is a test3", '刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
        list_of_docs2 = ["that is test4", "that is a test5", "that is a test6", '刘若英个演员', '唱歌很好听', 'men喜欢这首歌']
        corpus_new = [i + str(id) for id, i in enumerate(list_of_docs * 10)]
        m = HnswlibSimilarity(corpus=list_of_docs * 10)
        m.add_corpus(list_of_docs2)
        m.add_corpus(corpus_new)
        m.build_index()
        print(m.most_similar("This is a test4"))
        print(m.most_similar("men喜欢这首歌"))

        m.save_index('test1.model')
        m.load_index('test1.model')
        print(m.most_similar("This is a test4"))
        r = m.most_similar("men喜欢这首歌", topn=1)
        print(r)
        self.assertTrue(len(r[0]) == 1)
        os.remove('test1.model')

    def test_annoy_model(self):
        list_of_docs = ["This is a test1", "This is a test2", "This is a test3", '刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
        list_of_docs2 = ["that is test4", "that is a test5", "that is a test6", '刘若英个演员', '唱歌很好听', 'men喜欢这首歌']
        corpus_new = [i + str(id) for id, i in enumerate(list_of_docs * 10)]
        m = AnnoySimilarity(corpus=list_of_docs * 10)
        print(m)
        v = m._get_vector("This is test1")
        print(v[:10], v.shape)
        print(m.similarity("This is a test1", "that is a test5"))
        print(m.distance("This is a test1", "that is a test5"))
        print(m.most_similar("This is a test4"))
        print(m.most_similar("men喜欢这首歌"))
        m.add_corpus(list_of_docs2)
        m.add_corpus(corpus_new)
        m.build_index()
        print(m.most_similar("This is a test4"))
        r = m.most_similar("men喜欢这首歌", topn=1)
        print(r)
        self.assertTrue(len(r[0]) == 1)


if __name__ == '__main__':
    unittest.main()
