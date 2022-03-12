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
from PIL import Image
sys.path.append('..')
from similarities import ClipSimilarity, ImageHashSimilarity

pwd_path = os.path.abspath(os.path.dirname(__file__))
image_dir = os.path.join(pwd_path, '../examples/data/')

imgs = [Image.open(i) for i in glob(os.path.join(image_dir, '*.png'))]


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
        r = m.most_similar(imgs[:size], topn=5)
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
        r = m.most_similar(imgs[:size], topn=5)
        # print(r)
        spend_time = time() - t1
        print('[search] spend time:', spend_time, ' seconds, count:', size, ', qps:', size / spend_time)


if __name__ == '__main__':
    unittest.main()
