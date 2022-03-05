# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import unittest

sys.path.append('..')
from text2vec import SentenceModel
from similarities.semanticsim import BertSimilarity

sm = SentenceModel()
bert_model = BertSimilarity(sm)


class IssueTestCase(unittest.TestCase):

    def test_sim_diff(self):
        a = '研究团队面向国家重大战略需求追踪国际前沿发展借鉴国际人工智能研究领域的科研模式有效整合创新资源解决复'
        b = '英汉互译比较语言学'
        r = bert_model.similarity(a, b)
        print(a, b, r)
        self.assertTrue(abs(r - 0.1733) < 0.001)


if __name__ == '__main__':
    unittest.main()
