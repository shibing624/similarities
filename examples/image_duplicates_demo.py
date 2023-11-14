# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

from PIL import Image

sys.path.append('..')
from similarities import ClipSimilarity, paraphrase_mining_embeddings


def load_data(file_path):
    data_paths = []
    c = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            c += 1
            if c == 1:
                continue
            if line:
                path = line.split(',')[0]
                data_paths.append(path)
    return data_paths


def main():
    corpus_paths = load_data('data/image_info.csv')
    corpus_paths = list(set(corpus_paths))
    print('corpus size:', len(corpus_paths), 'top3:', corpus_paths[:3])

    model = ClipSimilarity()
    print(model)
    corpus = [Image.open(i) for i in corpus_paths]
    corpus_embeddings = model.get_embeddings(corpus, show_progress_bar=True, convert_to_tensor=True)
    duplicates = paraphrase_mining_embeddings(corpus_embeddings)

    for score, idx1, idx2 in duplicates[0:10]:
        print("\nScore: {:.3f}".format(score))
        print(corpus_paths[idx1])
        print(corpus_paths[idx2])

    # Score: 0.945
    # data/image1.png
    # data/image12-like-image1.png
    #
    # Score: 0.944
    # data/image10.png
    # data/image11-like-image10.png
    #
    # Score: 0.932
    # data/image8-like-image1.png
    # data/image12-like-image1.png


if __name__ == '__main__':
    main()
