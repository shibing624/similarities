# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This example starts multiple processes (1 per GPU), which encode
CLIP embeddings in parallel. This gives a near linear speed-up
when encoding large image collections.

This basic example loads a pre-trained model from the web and uses it to
generate embeddings for a given list of sentences.
"""
import sys

from PIL import Image

sys.path.append('..')
from similarities import ClipModule


def main():
    # 1. Compute embeddings for sentences
    # Create a large list of sentences
    sentences = ["This is sentence {}".format(i) for i in range(10000)]
    model = ClipModule("OFA-Sys/chinese-clip-vit-base-patch16")
    print(f"Sentences size: {len(sentences)}, model: {model}")

    # Start the multi processes pool on all available CUDA devices
    # target_devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
    pool = model.start_multi_process_pool(target_devices=None)

    # Compute the embeddings using the multi processes pool
    emb = model.encode_multi_process(sentences, pool, normalize_embeddings=True)
    print(f"Embeddings computed. Shape: {emb.shape}")

    # Optional: Stop the process in the pool
    model.stop_multi_process_pool(pool)

    # 2. Compute embeddings for images
    image_paths = ['data/image1.png'] * 100
    # convert to PIL images
    imgs = [Image.open(i) for i in image_paths]
    pool = model.start_multi_process_pool(target_devices=None)
    emb = model.encode_multi_process(imgs, pool, normalize_embeddings=True)
    print(f"Embeddings computed. Shape: {emb.shape}")
    model.stop_multi_process_pool(pool)


if __name__ == "__main__":
    main()
