# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Use Faiss for image similarity search demo
"""

import sys

import numpy as np

sys.path.append('..')
from similarities import clip_embedding, clip_index, clip_filter, clip_server


def main():
    # Build embedding
    clip_embedding(
        input_data_or_path='data/image_info.csv',
        columns=None,
        header=0,
        delimiter=',',
        image_embeddings_dir='tmp_image_embeddings_dir/',
        text_embeddings_dir=None,
        embeddings_name='emb.npy',
        corpus_file='tmp_data_dir/corpus.csv',
        model_name="OFA-Sys/chinese-clip-vit-base-patch16",
        batch_size=64,
        enable_image=True,
        enable_text=False,
        target_devices=['cuda:0', 'cuda:1'],
        normalize_embeddings=True,
    )

    # Build index
    clip_index(
        image_embeddings_dir='tmp_image_embeddings_dir/',
        text_embeddings_dir=None,
        image_index_dir='tmp_image_index_dir/',
        text_index_dir=None,
        index_name='faiss.index',
        max_index_memory_usage='1G',
        current_memory_available='2G',
        use_gpu=False,
        nb_cores=None,
    )

    # Filter(search) 文搜图, support multi query, batch search
    sentences = ['老虎', '花朵']
    clip_filter(
        texts=sentences,
        output_file=f"tmp_image_outputs/result_txt.json",
        model_name="OFA-Sys/chinese-clip-vit-base-patch16",
        index_dir='tmp_image_index_dir/',
        index_name="faiss.index",
        corpus_file="tmp_data_dir/corpus.csv",
        num_results=5,
        threshold=None,
        device=None,
    )

    # Filter(search) 图搜图, support multi query, batch search
    images = ['data/image1.png', 'data/image3.png']
    clip_filter(
        images=images,
        output_file=f"tmp_image_outputs/result_img.json",
        model_name="OFA-Sys/chinese-clip-vit-base-patch16",
        index_dir='tmp_image_index_dir/',
        index_name="faiss.index",
        corpus_file="tmp_data_dir/corpus.csv",
        num_results=5,
        threshold=None,
        device=None,
    )

    # Filter(search) 向量搜图, support multi query, batch search
    clip_filter(
        embeddings=np.random.randn(1, 512),
        output_file=f"tmp_image_outputs/result_img.json",
        model_name="OFA-Sys/chinese-clip-vit-base-patch16",
        index_dir='tmp_image_index_dir/',
        index_name="faiss.index",
        corpus_file="tmp_data_dir/corpus.csv",
        num_results=5,
        threshold=None,
        device=None,
    )

    # Start Server
    clip_server(
        model_name="OFA-Sys/chinese-clip-vit-base-patch16",
        index_dir='tmp_image_index_dir/',
        index_name="faiss.index",
        corpus_file="tmp_data_dir/corpus.csv",
        num_results=5,
        threshold=None,
        device=None,
        port=8002,
        debug=True,
    )


if __name__ == '__main__':
    main()
