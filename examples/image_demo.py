# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import glob

sys.path.append('..')
from similarities.imagesim import ImageHashSimilarity, SiftSimilarity, ClipSimilarity


def clip_demo():
    m = ClipSimilarity()
    print(m)
    sim_scores = m.similarity(image_fps1, image_fps2)
    print('sim scores: ', sim_scores)
    for (idx, i), j in zip(enumerate(image_fps1), image_fps2):
        s = sim_scores[idx][idx]
        print(f"{i} vs {j}, score: {s:.4f}")
    # search
    m.add_corpus(corpus)
    queries = image_fps1
    search_res = m.most_similar(queries, topn=3)
    print('sim search: ', search_res)
    for q, r in zip(queries, search_res):
        print(f"query: {q}\t search result: {r}")
    print('-' * 50 + '\n')


def phash_demo():
    m = ImageHashSimilarity(hash_function='phash')
    print(m)
    sim_scores = m.similarity(image_fps1, image_fps2)
    print('sim scores: ', sim_scores)
    for i, j, s in zip(image_fps1, image_fps2, sim_scores):
        print(f"{i} vs {j}, score: {s:.4f}")

    m.add_corpus(corpus)
    queries = image_fps1
    search_res = m.most_similar(queries, topn=3)
    print('sim search: ', search_res)
    for q, r in zip(queries, search_res):
        print(f"query: {q}\t search result: {r}")
    print('-' * 50 + '\n')


def sift_demo():
    m = SiftSimilarity()
    print(m)
    sim_scores = m.similarity(image_fps1, image_fps2)
    print('sim scores: ', sim_scores)
    for i, j, s in zip(image_fps1, image_fps2, sim_scores):
        print(f"{i} vs {j}, score: {s:.4f}")

    m.add_corpus(corpus)
    queries = image_fps1
    search_res = m.most_similar(queries, topn=3)
    print('sim search: ', search_res)
    for q, r in zip(queries, search_res):
        print(f"query: {q}\t search result: {r}")
    print('-' * 50 + '\n')


if __name__ == "__main__":
    image_fps1 = ['data/image1.png', 'data/image3.png']
    image_fps2 = ['data/image12-like-image1.png', 'data/image10.png']
    corpus = glob.glob('data/*.jpg') + glob.glob('data/*.png')

    clip_demo()  # the best result
    phash_demo()
    sift_demo()
