# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import glob

sys.path.append('..')
from similarities.imagesim import ImageHashSimilarity, SiftSimilarity, ClipSimilarity


def phash_demo(image_fp1, image_fp2):
    m = ImageHashSimilarity(hash_function='phash')
    print(m)
    print(m.similarity(image_fp1, image_fp2))
    m.most_similar(image_fp1)
    # no corpus
    m.add_corpus(glob.glob('data/*.jpg') + glob.glob('data/*.png'))
    r = m.most_similar(image_fp1)
    print(r)

    m = ImageHashSimilarity(hash_function='average_hash')
    print(m)
    print(m.similarity(image_fp1, image_fp2))
    m.most_similar(image_fp1)
    # no corpus
    m.add_corpus(glob.glob('data/*.jpg') + glob.glob('data/*.png'))
    r = m.most_similar(image_fp1)
    print(r)


def sift_demo(image_fp1, image_fp2):
    m = SiftSimilarity()
    print(m)
    print(m.similarity(image_fp1, image_fp2))
    m.most_similar(image_fp1)
    # no corpus
    m.add_corpus(glob.glob('data/*.jpg'))
    m.add_corpus(glob.glob('data/*.png'))
    r = m.most_similar(image_fp1)
    print(r)


def clip_demo(image_fp1, image_fp2):
    m = ClipSimilarity()
    print(m)
    print(m.similarity(image_fp1, image_fp2))
    m.most_similar(image_fp1)
    # no corpus
    m.add_corpus(glob.glob('data/*.jpg') + glob.glob('data/*.png'))
    r = m.most_similar(image_fp1)
    print(r)


if __name__ == "__main__":
    image_fp1 = 'data/image1.png'
    image_fp2 = 'data/image12-like-image1.png'

    phash_demo(image_fp1, image_fp2)
    sift_demo(image_fp1, image_fp2)
    clip_demo(image_fp1, image_fp2)
