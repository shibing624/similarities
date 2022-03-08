# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import unittest

sys.path.append('..')
from text2vec import SentenceModel
from similarities.similarity import Similarity

sm = SentenceModel()
m = Similarity(sm)


class SimScoreTestCase(unittest.TestCase):

    def test_sim_diff(self):
        a = '研究团队面向国家重大战略需求追踪国际前沿发展借鉴国际人工智能研究领域的科研模式有效整合创新资源解决复'
        b = '英汉互译比较语言学'
        r = m.similarity(a, b)
        print(a, b, r)
        self.assertTrue(abs(r - 0.1733) < 0.001)

    def test_empty(self):
        v = m._get_vector("This is test1")
        print(v[:10], v.shape)
        print(m.similarity("This is a test1", "that is a test5"))
        print(m.distance("This is a test1", "that is a test5"))
        print(m.most_similar("This is a test4"))


if __name__ == '__main__':
    unittest.main()
