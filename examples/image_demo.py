# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import glob

sys.path.append('..')
from similarities.imagesim import ImageHashSimilarity, SiftSimilarity, ClipSimilarity


def sim_and_search(m):
    print(m)
    # similarity
    sim_scores = m.similarity(image_fps1, image_fps2)
    print('sim scores: ', sim_scores)
    for (idx, i), j in zip(enumerate(image_fps1), image_fps2):
        s = sim_scores[idx] if isinstance(sim_scores, list) else sim_scores[idx][idx]
        print(f"{i} vs {j}, score: {s:.4f}")
    # search
    m.add_corpus(corpus)
    queries = image_fps1
    res = m.most_similar(queries, topn=3)
    print('sim search: ', res)
    for q_id, c in res.items():
        print('query:', queries[q_id])
        print("search top 3:")
        for corpus_id, s in c.items():
            print(f'\t{m.corpus[corpus_id]}: {s:.4f}')
    print('-' * 50 + '\n')


if __name__ == "__main__":
    image_fps1 = ['data/image1.png', 'data/image3.png']
    image_fps2 = ['data/image12-like-image1.png', 'data/image10.png']
    corpus = glob.glob('data/*.jpg') + glob.glob('data/*.png')

    sim_and_search(ClipSimilarity())  # the best result
    sim_and_search(ImageHashSimilarity(hash_function='phash'))
    sim_and_search(SiftSimilarity())
