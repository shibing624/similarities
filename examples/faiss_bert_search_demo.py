# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Use Faiss for text similarity search demo
"""

import sys

sys.path.append('..')
from similarities.faiss_bert_similarity import bert_embedding, bert_index, bert_filter


def main():
    # Build embedding
    bert_embedding(
        input_dir='data/toy_corpus/',
        embeddings_dir='tmp_embeddings_dir/',
        embeddings_name='emb.npy',
        corpus_file='tmp_data_dir/corpus.npy',
        model_name="shibing624/text2vec-base-chinese",
        batch_size=12,
        device=None
    )

    # Build index
    bert_index(
        embeddings_dir='tmp_embeddings_dir/',
        index_dir="tmp_index_dir/",
        index_name="faiss.index",
        max_index_memory_usage="1G",
        current_memory_available="2G",
        use_gpu=False,
        nb_cores=None,
    )

    # Filter(search)
    sentences = ['如何更换花呗绑定银行卡',
                 '花呗更改绑定银行卡']
    for i, q in enumerate(sentences):
        bert_filter(
            query=q,
            output_file=f"tmp_outputs/result_{i}.json",
            model_name="shibing624/text2vec-base-chinese",
            index_dir='tmp_index_dir/',
            index_name="faiss.index",
            corpus_file="tmp_data_dir/corpus.npy",
            num_results=10,
            threshold=None,
            device=None,
        )


if __name__ == '__main__':
    main()
