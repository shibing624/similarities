# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
import unittest

sys.path.append('..')

from similarities.literalsim import SimHashSimilarity, TfidfSimilarity, BM25Similarity, WordEmbeddingSimilarity, \
    CilinSimilarity, HownetSimilarity

from similarities.utils.distance import hamming_distance


class LiteralCase(unittest.TestCase):
    def test_hamming_distance(self):
        text1 = '刘若英是个演员'
        text2 = '他唱歌很好听'
        m = SimHashSimilarity()
        seq1 = m.simhash(text1)
        seq2 = m.simhash(text2)
        print(seq1)
        print(seq2)
        r = 1.0 - hamming_distance(seq1, seq2) / 64
        print(hamming_distance(seq1, seq2))
        print(r)
        print(m.similarity(text1, text2))

        text1 = '刘若英是个演员'
        text2 = '他'
        m = SimHashSimilarity()
        seq1 = m.simhash(text1)
        seq2 = m.simhash(text2)
        print(seq1)
        print(seq2)
        print(m.similarity(text1, text2))

        text1 = '刘若'
        text2 = '他'
        m = SimHashSimilarity()
        seq1 = m.simhash(text1)
        seq2 = m.simhash(text2)
        print(seq1)
        print(seq2)
        print(m.similarity(text1, text2))

        text1 = '刘若'
        text2 = '他他唱歌很好听他唱歌很好听他唱歌很好听他唱歌很好听他唱歌很好听他唱歌很好听,他唱歌很好听?他唱歌很好听？他唱歌很好听。。'
        m = SimHashSimilarity()
        seq1 = m.simhash(text1)
        seq2 = m.simhash(text2)
        print(seq1)
        print(seq2)
        print(m.similarity(text1, text2))

        text1 = '刘若 他唱歌很好听他唱歌很好听他唱歌很好听他唱歌很好听他唱歌很好听他唱歌很好听,他唱歌很好听?他唱歌很好听？他唱歌很好'
        text2 = '他他唱歌很好听他唱歌很好听他唱歌很好听他唱歌很好听他唱歌很好听他唱歌很好听,他唱歌很好听?他唱歌很好听？他唱歌很好听。。'
        m = SimHashSimilarity()
        seq1 = m.simhash(text1)
        seq2 = m.simhash(text2)
        print(seq1)
        print(seq2)
        s = m.similarity(text1, text2)
        print(s)
        self.assertTrue(s[0] > 0.5)

    def test_simhash(self):
        """test_simhash"""
        text1 = '刘若英是个演员'
        text2 = '他唱歌很好听'
        m = SimHashSimilarity()
        print(m.similarity(text1, text2))
        print(m.distance(text1, text2))
        r = m.most_similar('刘若英是演员')
        self.assertEqual(len(r[0]), 0)
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
        m.add_corpus(zh_list)
        r = m.most_similar('刘若英是演员', topn=2)
        print(r)
        self.assertEqual(len(r[0]), 2)

    def test_tfidf(self):
        """test_tfidf"""
        text1 = '刘若英是个演员'
        text2 = '他唱歌很好听'
        m = TfidfSimilarity()
        print(m.similarity(text1, text2))
        print(m.distance(text1, text2))
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌', '我不是演员吗']
        m.add_corpus(zh_list)
        r = m.most_similar('刘若英是演员')
        print(r)
        self.assertEqual(len(r[0]), 4)
        r = m.most_similar(['刘若英是演员', '唱歌很好听'])
        print(r)
        self.assertEqual(len(r), 2)

    def test_bm25(self):
        """test_bm25"""
        text1 = '刘若英是个演员'
        text2 = '他唱歌很好听'
        m = BM25Similarity()
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌', '我不是演员吗']
        m.add_corpus(zh_list)
        r = m.most_similar('刘若英是演员', topn=10)
        print(r)
        self.assertEqual(len(r[0]), 4)
        r = m.most_similar(['刘若英是演员', '唱歌很好听'])
        print(r)
        self.assertEqual(len(r), 2)

    def test_word2vec(self):
        """test_word2vec"""
        text1 = '刘若英是个演员'
        text2 = '他唱歌很好听'
        list_of_corpus = ["This is a test1", "This is a test2", "This is a test3"]
        list_of_corpus2 = ["that is test4", "that is a test5", "that is a test6"]
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌', '刘若英是个演员', '演戏很好看的人']
        m = WordEmbeddingSimilarity(list_of_corpus)
        print(m.similarity(text1, text2))
        print(m.distance(text1, text2))
        m.add_corpus(list_of_corpus2 + zh_list)
        v = m._get_vector("This is a test1")
        print(v[:10], v.shape)
        r = m.most_similar('刘若英是演员', topn=4)
        print(r)
        self.assertEqual(len(r[0]), 4)
        r = m.most_similar(['刘若英是演员', '唱歌很好听'])
        print(r)
        self.assertEqual(len(r), 2)

    def test_cilin(self):
        """test_cilin"""
        text1 = '周杰伦是一个歌手'
        text2 = '刘若英是个演员'
        m = CilinSimilarity()
        print(m.similarity(text1, text2))
        print(m.distance(text1, text2))
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
        m.add_corpus(zh_list)
        r = m.most_similar('刘若英是演员', topn=3)
        print(r)
        self.assertEqual(len(r[0]), 3)
        r = m.most_similar(['刘若英是演员', '唱歌很好听'])
        print(r)
        self.assertEqual(len(r), 2)

    def test_hownet(self):
        """test_cilin"""
        text1 = '周杰伦是一个歌手'
        text2 = '刘若英是个演员'
        m = HownetSimilarity()
        print(m.similarity(text1, text2))
        print(m.distance(text1, text2))
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
        m.add_corpus(zh_list)
        r = m.most_similar('刘若英是演员', topn=2)
        print(r)
        self.assertEqual(len(r[0]), 2)
        r = m.most_similar(['刘若英是演员', '唱歌很好听'])
        print(r)
        self.assertEqual(len(r), 2)


if __name__ == '__main__':
    unittest.main()
