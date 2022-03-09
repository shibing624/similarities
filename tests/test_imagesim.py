# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import glob
import os
import sys
import unittest

sys.path.append('..')

from similarities.imagesim import ClipSimilarity, ImageHashSimilarity, SiftSimilarity

pwd_path = os.path.abspath(os.path.dirname(__file__))

image_fp1 = os.path.join(pwd_path, '../examples/data/image1.png')
image_fp2 = os.path.join(pwd_path, '../examples/data/image8-like-image1.png')
image_dir = os.path.join(pwd_path, '../examples/data/')


class ImageSimCase(unittest.TestCase):
    def test_clip(self):
        m = ClipSimilarity(glob.glob(f'{image_dir}/*.jpg'))
        print(m)
        s = m.similarity(image_fp1, image_fp2)
        print(s)
        self.assertTrue(s > 0.5)
        r = m.most_similar(image_fp1)
        print(r)
        self.assertTrue(not r[0])
        # no corpus
        m.add_corpus(glob.glob(f'{image_dir}/*.jpg'))
        m.add_corpus(glob.glob(f'{image_dir}/*.png'))

        r = m.most_similar(image_fp1)
        print(r)
        self.assertTrue(len(r) > 0)

    def test_sift(self):
        m = SiftSimilarity(corpus=glob.glob(f'{image_dir}/*.jpg'))
        print(m)
        print(m.similarity(image_fp1, image_fp2))
        r = m.most_similar(image_fp1)
        print(r)
        self.assertTrue(not r[0])
        # no corpus
        m.add_corpus(glob.glob(f'{image_dir}/*.jpg'))
        m.add_corpus(glob.glob(f'{image_dir}/*.png'))
        m.add_corpus(glob.glob(f'{image_dir}/*.png'))
        r = m.most_similar(image_fp1)
        print(r)
        self.assertTrue(len(r) > 0)

    def test_phash(self):
        m = ImageHashSimilarity(hash_function='phash', corpus=glob.glob(f'{image_dir}/*.jpg'))
        print(m)
        print(m.similarity(image_fp1, image_fp2))
        m.most_similar(image_fp1)
        # no corpus
        m.add_corpus(glob.glob(f'{image_dir}/*.jpg') + glob.glob(f'{image_dir}/*.png'))
        r = m.most_similar(image_fp1)
        print(r)

        m = ImageHashSimilarity(hash_function='average_hash', corpus=glob.glob(f'{image_dir}/*.jpg'))
        print(m)
        print(m.similarity(image_fp1, image_fp2))
        m.most_similar(image_fp1)
        # no corpus
        m.add_corpus(glob.glob(f'{image_dir}/*.png'))
        m.add_corpus(glob.glob(f'{image_dir}/*.png'))
        r = m.most_similar(image_fp1)
        print(r)
        self.assertTrue(len(r) > 0)

    def test_hamming_distance(self):
        m = ImageHashSimilarity(hash_function='phash', hash_size=128)
        print(m.similarity(image_fp1, image_fp2))
        image_fp3 = os.path.join(pwd_path, '../examples/data/image3.png')

        s = m.similarity(image_fp1, image_fp3)
        print(s)
        self.assertTrue(s[0] > 0)


if __name__ == '__main__':
    unittest.main()
