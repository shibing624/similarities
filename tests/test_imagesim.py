# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import glob
import os
import sys
import unittest
from PIL import Image

sys.path.append('..')

from similarities.imagesim import ClipSimilarity, ImageHashSimilarity, SiftSimilarity

pwd_path = os.path.abspath(os.path.dirname(__file__))

img1 = Image.open(os.path.join(pwd_path, '../examples/data/image1.png'))
img2 = Image.open(os.path.join(pwd_path, '../examples/data/image8-like-image1.png'))
image_dir = os.path.join(pwd_path, '../examples/data/')
corpus_imgs = [Image.open(i) for i in glob.glob(os.path.join(image_dir, '*.png'))]


class ImageSimCase(unittest.TestCase):

    def test_clip(self):
        m = ClipSimilarity()
        print(m)
        s = m.similarity(img1, img2)
        print(s)
        self.assertTrue(s > 0.5)
        r = m.most_similar(img1)
        print(r)
        self.assertTrue(not r[0])
        m.add_corpus(corpus_imgs)

        r = m.most_similar(img1)
        print(r)
        self.assertTrue(len(r) > 0)

    def test_clip_dict(self):
        m = ClipSimilarity()
        print(m)
        corpus_dict = {i.filename: i for i in corpus_imgs}
        queries = {i.filename: i for i in corpus_imgs[:3]}
        m.add_corpus(corpus_dict)
        r = m.most_similar(queries)
        print(r)
        self.assertTrue(len(r) > 0)

    def test_sift(self):
        m = SiftSimilarity(corpus=glob.glob(f'{image_dir}/*.jpg'))
        print(m)
        print(m.similarity(img1, img2))
        r = m.most_similar(img1)
        print(r)
        self.assertTrue(not r[0])
        m.add_corpus(corpus_imgs)
        m.add_corpus(corpus_imgs)
        r = m.most_similar(img1)
        print(r)
        self.assertTrue(len(r) > 0)

    def test_phash(self):
        m = ImageHashSimilarity(hash_function='phash')
        print(m)
        print(m.similarity(img1, img2))
        m.most_similar(img1)
        m.add_corpus(corpus_imgs)
        r = m.most_similar(img1)
        print(r)

        m = ImageHashSimilarity(hash_function='average_hash')
        print(m)
        print(m.similarity(img1, img2))
        m.most_similar(img1)
        m.add_corpus(corpus_imgs)
        m.add_corpus(corpus_imgs)
        r = m.most_similar(img1)
        print(r)
        self.assertTrue(len(r) > 0)

    def test_hamming_distance(self):
        m = ImageHashSimilarity(hash_function='phash', hash_size=128)
        s = m.similarity(img1, img2)
        print(s)
        self.assertTrue(s[0] > 0)


if __name__ == '__main__':
    unittest.main()
