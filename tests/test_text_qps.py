# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys
import unittest
from time import time

sys.path.append('..')
from similarities import SimHashSimilarity, TfidfSimilarity, BM25Similarity, WordEmbeddingSimilarity, \
    CilinSimilarity, HownetSimilarity, Similarity, HnswlibSimilarity

pwd_path = os.path.abspath(os.path.dirname(__file__))
sts_test_path = os.path.join(pwd_path, 'test.txt')


def load_test_data(path):
    sents1, sents2, labels = [], [], []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) != 3:
                continue
            sents1.append(line[0])
            sents2.append(line[1])
            labels.append(int(line[2]))
            if len(sents1) > 500:
                break
    return sents1, sents2, labels


sents1, sents2, labels = load_test_data(sts_test_path)


class QPSSimTestCase(unittest.TestCase):
    def test_sim_speed(self):
        """test_sim_speed"""
        m = Similarity()
        print(m)
        t1 = time()
        r = m.similarity(sents1, sents2)
        print(r[:10])
        print(labels[:10])
        spend_time = time() - t1
        print('[sim] spend time:', spend_time, ' seconds, count:', len(sents2), ', qps:', len(sents2) / spend_time)
        m.add_corpus(sents2)
        t1 = time()
        size = 100
        r = m.most_similar(sents1[:size], topn=5)
        # print(r)
        spend_time = time() - t1
        print('[search] spend time:', spend_time, ' seconds, count:', size, ', qps:', size / spend_time)

    def test_hnsw_speed(self):
        m = HnswlibSimilarity()
        print(m)
        t1 = time()
        a = sents1[:100]
        b = sents2[:100]
        r = m.similarity(a, b)
        print(r[:10])
        print(labels[:10])
        spend_time = time() - t1
        print('[sim] spend time:', spend_time, ' seconds, count:', len(a), ', qps:', len(a) / spend_time)
        m.add_corpus(b)
        t1 = time()
        size = 100
        r = m.most_similar(sents1[:size], topn=5)
        # print(r)
        spend_time = time() - t1
        print('[search] spend time:', spend_time, ' seconds, count:', size, ', qps:', size / spend_time)

    def test_w2v_speed(self):
        m = WordEmbeddingSimilarity()
        print(m)
        t1 = time()
        r = m.similarity(sents1, sents2)
        print(r[:10])
        print(labels[:10])
        spend_time = time() - t1
        print('[sim] spend time:', spend_time, ' seconds, count:', len(sents2), ', qps:', len(sents2) / spend_time)
        m.add_corpus(sents2)
        t1 = time()
        size = 100
        r = m.most_similar(sents1[:size], topn=5)
        # print(r)
        spend_time = time() - t1
        print('[search] spend time:', spend_time, ' seconds, count:', size, ', qps:', size / spend_time)

    def test_simhash_speed(self):
        m = SimHashSimilarity()
        print(m)
        t1 = time()
        r = m.similarity(sents1, sents2)
        print(r[:10])
        print(labels[:10])
        spend_time = time() - t1
        print('[sim] spend time:', spend_time, ' seconds, count:', len(sents2), ', qps:', len(sents2) / spend_time)
        m.add_corpus(sents2)
        t1 = time()
        size = 100
        r = m.most_similar(sents1[:size], topn=5)
        # print(r)
        spend_time = time() - t1
        print('[search] spend time:', spend_time, ' seconds, count:', size, ', qps:', size / spend_time)

    def test_tfidf_speed(self):
        m = TfidfSimilarity()
        print(m)
        t1 = time()
        a = sents1[:100]
        b = sents2[:100]
        r = m.similarity(a, b)
        for i in range(len(a)):
            print(r[i][i], labels[i])
        spend_time = time() - t1
        print('[sim] spend time:', spend_time, ' seconds, count:', len(a), ', qps:', len(a) / spend_time)
        m.add_corpus(sents2)
        t1 = time()
        size = 100
        r = m.most_similar(sents1[:size], topn=5)
        # print(r)
        spend_time = time() - t1
        print('[search] spend time:', spend_time, ' seconds, count:', size, ', qps:', size / spend_time)

    def test_bm25_speed(self):
        m = BM25Similarity()
        print(m)
        m.add_corpus(sents2)
        t1 = time()
        size = 100
        r = m.most_similar(sents1[:size], topn=5)
        # print(r)
        spend_time = time() - t1
        print('[search] spend time:', spend_time, ' seconds, count:', size, ', qps:', size / spend_time)

    def test_cilin_speed(self):
        m = CilinSimilarity()
        print(m)
        t1 = time()
        a = sents1[:100]
        b = sents2[:100]
        r = m.similarity(a, b)
        for i in range(len(a)):
            print(r[i], labels[i])
        spend_time = time() - t1
        print('[sim] spend time:', spend_time, ' seconds, count:', len(a), ', qps:', len(a) / spend_time)
        m.add_corpus(sents2)
        t1 = time()
        size = 20
        r = m.most_similar(sents1[:size], topn=5)
        # print(r)
        spend_time = time() - t1
        print('[search] spend time:', spend_time, ' seconds, count:', size, ', qps:', size / spend_time)
        self.assertTrue(len(r) > 0)


if __name__ == '__main__':
    unittest.main()
