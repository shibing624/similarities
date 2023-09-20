# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Use Faiss for image similarity search demo
"""

import sys

import numpy as np

sys.path.append('..')
from similarities import ClipClient, ClipItem


def main():
    # Client
    client = ClipClient('http://0.0.0.0:8002')

    # 获取嵌入，支持获取文本嵌入、图片嵌入
    text_input = "This is a sample text."
    emb = client.get_emb(text=text_input)
    print(f"Embedding for '{text_input}': {emb}")
    # input image
    image_input = "data/image1.png"
    emb = client.get_emb(image=image_input)
    print(f"Embedding for '{image_input}': {emb}")

    # 获取相似度，支持计算图文相似度、图片相似度
    item1 = ClipItem(image="data/image1.png")
    item2 = ClipItem(text="老虎")
    similarity = client.get_similarity(item1, item2)
    print(f"Similarity between item1 and item2: {similarity}")

    # 搜索
    # 1. 文搜图
    search_input = "This is a sample text."
    search_results = client.search(text=search_input)
    print(f"Search results for '{search_input}': {search_results}")
    # 2. 图搜图
    search_input = "data/image1.png"
    search_results = client.search(image=search_input)
    print(f"Search results for '{search_input}': {search_results}")
    # 3. 向量搜图
    search_results = client.search(emb=np.random.randn(512).tolist())
    print(f"Search results for emb search: {search_results}")


if __name__ == '__main__':
    main()
