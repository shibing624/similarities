# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys

from PIL import Image

sys.path.append('..')
from similarities import ClipModule


def main():
    # image paths
    image_paths = ['data/image1.png', 'data/image3.png', 'data/image5.png']
    model = ClipModule("OFA-Sys/chinese-clip-vit-base-patch16")
    print(f"data size: {len(image_paths)}")
    # convert to PIL images
    imgs = [Image.open(i) for i in image_paths]
    emb = model.encode(imgs, normalize_embeddings=True)
    print(f"Embeddings computed. Shape: {emb.shape}")


if __name__ == "__main__":
    main()
