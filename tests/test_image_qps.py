# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys
import unittest
from time import time
from glob import glob

sys.path.append('..')
from similarities import *

pwd_path = os.path.abspath(os.path.dirname(__file__))
img_dir = os.path.join(pwd_path, '../examples/data/')

imgs = glob(f'{img_dir}/*.png')


class QPSImageTestCase(unittest.TestCase):
    def test_clip_sim_speed(self):
        """test_clip_sim_speed"""
        m = ClipSimilarity()
        t1 = time()
        size = 5
        r = m.similarity(imgs[0:size], imgs[1:1 + size])
        print(r)
        spend_time = time() - t1
        print('[sim] spend time:', spend_time, ' seconds, count:', size, ', qps:', size / spend_time)
        m.add_corpus(imgs)
        t1 = time()
        size = 10
        for q in imgs[:size]:
            r = m.most_similar(q, topn=5)
            # print(r)
        spend_time = time() - t1
        print('[search] spend time:', spend_time, ' seconds, count:', size, ', qps:', size / spend_time)

    def test_phash_speed(self):
        m = ImageHashSimilarity()
        t1 = time()
        size = 5
        r = m.similarity(imgs[0:size], imgs[1:1 + size])
        print(r)
        spend_time = time() - t1
        print('[sim] spend time:', spend_time, ' seconds, count:', size, ', qps:', size / spend_time)
        m.add_corpus(imgs)
        t1 = time()
        size = 10
        for q in imgs[:size]:
            r = m.most_similar(q, topn=5)
            # print(r)
        spend_time = time() - t1
        print('[search] spend time:', spend_time, ' seconds, count:', size, ', qps:', size / spend_time)


if __name__ == '__main__':
    unittest.main()
