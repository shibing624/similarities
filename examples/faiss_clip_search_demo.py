# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Use Faiss for image similarity search demo
"""

import sys

sys.path.append('..')
from similarities.faiss_clip_similarity import clip_embedding, clip_index, clip_filter


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
        batch_size=12,
        enable_image=True,
        enabel_text=False,
        device=None
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

    # Filter(search) 文搜图
    sentences = ['老虎', '花朵']
    for i, q in enumerate(sentences):
        clip_filter(
            query=q,
            output_file=f"tmp_image_outputs/result_txt_{i}.json",
            model_name="OFA-Sys/chinese-clip-vit-base-patch16",
            index_dir='tmp_image_index_dir/',
            index_name="faiss.index",
            corpus_file="tmp_data_dir/corpus.csv",
            num_results=5,
            threshold=None,
            device=None,
        )

    # Filter(search) 图搜图
    images = ['data/image1.png', 'data/image10.png']
    for i, q in enumerate(images):
        clip_filter(
            query=q,
            output_file=f"tmp_image_outputs/result_img_{i}.json",
            model_name="OFA-Sys/chinese-clip-vit-base-patch16",
            index_dir='tmp_image_index_dir/',
            index_name="faiss.index",
            corpus_file="tmp_data_dir/corpus.csv",
            num_results=5,
            threshold=None,
            device=None,
        )


if __name__ == '__main__':
    main()
