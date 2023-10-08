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
        input_dir='data/toy_clip/',
        chunk_size=10000,
        image_embeddings_dir='clip_engine/image_emb/',
        text_embeddings_dir=None,
        corpus_dir='clip_engine/corpus/',
        model_name="OFA-Sys/chinese-clip-vit-base-patch16",
        batch_size=12,
        enable_image=True,
        image_column_name='image_path',
        enable_text=False,
        target_devices=None,
        normalize_embeddings=True,
        header=0,
    )

    # Build index
    clip_index(
        image_embeddings_dir='clip_engine/image_emb/',
        text_embeddings_dir=None,
        image_index_dir='clip_engine/image_index/',
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
        output_file="tmp_outputs/result_txt.json",
        model_name="OFA-Sys/chinese-clip-vit-base-patch16",
        index_dir='clip_engine/image_index/',
        index_name="faiss.index",
        corpus_dir='clip_engine/corpus/',
        num_results=5,
        threshold=None,
        device=None,
    )

    # Filter(search) 图搜图, support multi query, batch search
    images = ['data/image1.png', 'data/image3.png']
    clip_filter(
        images=images,
        output_file="tmp_outputs/result_img.json",
        model_name="OFA-Sys/chinese-clip-vit-base-patch16",
        index_dir='clip_engine/image_index/',
        index_name="faiss.index",
        corpus_dir='clip_engine/corpus/',
        num_results=5,
        threshold=None,
        device=None,
    )

    # Filter(search) 向量搜图, support multi query, batch search
    clip_filter(
        embeddings=np.random.randn(1, 512),
        output_file="tmp_outputs/result_emb.json",
        model_name="OFA-Sys/chinese-clip-vit-base-patch16",
        index_dir='clip_engine/image_index/',
        index_name="faiss.index",
        corpus_dir='clip_engine/corpus/',
        num_results=5,
        threshold=None,
        device=None,
    )

    # Start Server
    clip_server(
        model_name="OFA-Sys/chinese-clip-vit-base-patch16",
        index_dir='clip_engine/image_index/',
        index_name="faiss.index",
        corpus_dir='clip_engine/corpus/',
        num_results=5,
        threshold=None,
        device=None,
        port=8002,
        debug=True,
    )


if __name__ == '__main__':
    main()
