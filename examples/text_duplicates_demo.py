# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Find duplicate sentences in a corpus
"""
import sys

sys.path.append('..')
from similarities import BertSimilarity, paraphrase_mining_embeddings


def load_data(file_path):
    corpus = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(line)
    return corpus


def main():
    corpus = load_data('data/corpus.txt')
    corpus = list(set(corpus))
    print('corpus size:', len(corpus), 'top3:', corpus[:3])

    model = BertSimilarity(model_name_or_path="shibing624/text2vec-base-chinese")
    print(model)
    corpus_embeddings = model.get_embeddings(corpus, show_progress_bar=True, convert_to_tensor=True)
    duplicates = paraphrase_mining_embeddings(corpus_embeddings)

    for score, idx1, idx2 in duplicates[0:10]:
        print("\nScore: {:.3f}".format(score))
        print(corpus[idx1])
        print(corpus[idx2])

    # Score: 0.996
    # 两条狗在草地上奔跑。
    # 两只狗在草地上奔跑。
    #
    # Score: 0.995
    # 一个女人在吹长笛。
    # 那个女人在吹长笛。
    #
    # Score: 0.994
    # 一只黑狗在水中奔跑。
    # 一只黑狗正在水中奔跑。


if __name__ == '__main__':
    main()
