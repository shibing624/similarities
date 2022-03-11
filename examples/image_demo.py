# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import glob
from PIL import Image

sys.path.append('..')
from similarities.imagesim import ImageHashSimilarity, SiftSimilarity, ClipSimilarity


def sim_and_search(m):
    print(m)
    # similarity
    sim_scores = m.similarity(imgs1, imgs2)
    print('sim scores: ', sim_scores)
    for (idx, i), j in zip(enumerate(image_fps1), image_fps2):
        s = sim_scores[idx] if isinstance(sim_scores, list) else sim_scores[idx][idx]
        print(f"{i} vs {j}, score: {s:.4f}")
    # search
    m.add_corpus(corpus_imgs)
    queries = imgs1
    res = m.most_similar(queries, topn=3)
    print('sim search: ', res)
    for q_id, c in res.items():
        print('query:', image_fps1[q_id])
        print("search top 3:")
        for corpus_id, s in c.items():
            print(f'\t{m.corpus[corpus_id].filename}: {s:.4f}')
    print('-' * 50 + '\n')


def clip_demo():
    m = ClipSimilarity()
    print(m)
    # similarity score between text and image
    image_fps = ['data/image3.png',  # yellow flower image
                 'data/image1.png']  # tiger image
    texts = ['a yellow flower', 'a tiger']
    imgs = [Image.open(i) for i in image_fps]
    sim_scores = m.similarity(imgs, texts)
    print('sim scores: ', sim_scores)
    for (idx, i), j in zip(enumerate(image_fps), texts):
        s = sim_scores[idx][idx]
        print(f"{i} vs {j}, score: {s:.4f}")
    print('-' * 50 + '\n')


if __name__ == "__main__":
    image_fps1 = ['data/image1.png', 'data/image3.png']
    image_fps2 = ['data/image12-like-image1.png', 'data/image10.png']
    imgs1 = [Image.open(i) for i in image_fps1]
    imgs2 = [Image.open(i) for i in image_fps2]
    corpus_fps = glob.glob('data/*.jpg') + glob.glob('data/*.png')
    corpus_imgs = [Image.open(i) for i in corpus_fps]
    # 1. image and text similarity
    clip_demo()

    # 2. image and image similarity score
    sim_and_search(ClipSimilarity())  # the best result
    sim_and_search(ImageHashSimilarity(hash_function='phash'))
    sim_and_search(SiftSimilarity())
