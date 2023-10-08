# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Use Faiss for text similarity search demo
"""

import sys

sys.path.append('..')
from similarities import bert_embedding, bert_index, bert_filter, bert_server


def main():
    # Build embedding
    bert_embedding(
        input_dir='data/toy_bert/',
        embeddings_dir='bert_engine/text_emb/',
        corpus_dir='bert_engine/corpus/',
        model_name="shibing624/text2vec-base-chinese",
        batch_size=12,
        target_devices=None,
        normalize_embeddings=True,
        text_column_name="sentence",
        header=None,
        names=['sentence'],
    )

    # Build index
    bert_index(
        embeddings_dir='bert_engine/text_emb/',
        index_dir="bert_engine/text_index/",
        index_name="faiss.index",
        max_index_memory_usage="1G",
        current_memory_available="2G",
        use_gpu=False,
        nb_cores=None,
    )

    # Filter(search) support multi query, batch search
    sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']
    bert_filter(
        queries=sentences,
        output_file=f"outputs/result.json",
        model_name="shibing624/text2vec-base-chinese",
        index_dir='bert_engine/text_index/',
        index_name="faiss.index",
        corpus_dir="bert_engine/corpus/",
        num_results=5,
        threshold=None,
        device=None,
    )

    # Server
    bert_server(
        model_name="shibing624/text2vec-base-chinese",
        index_dir='bert_engine/text_index/',
        index_name="faiss.index",
        corpus_dir="bert_engine/corpus/",
        num_results=5,
        threshold=None,
        device=None,
        port=8001,
    )


if __name__ == '__main__':
    main()
