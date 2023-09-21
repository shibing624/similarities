# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Use Faiss for text similarity search demo
"""

import sys

sys.path.append('..')
from similarities import BertClient


def main():
    # Client
    client = BertClient('http://0.0.0.0:8001')

    # 获取嵌入
    text_input = "This is a sample text."
    emb = client.get_emb(text_input)
    print(f"Embedding for '{text_input}': {emb}")

    # 获取相似度
    similarity = client.get_similarity("This is a sample text.", "This is another sample text.")
    print(f"Similarity between item1 and item2: {similarity}")

    # 搜索
    search_input = "This is a sample text."
    search_results = client.search(search_input)
    print(f"Search results for '{search_input}': {search_results}")


if __name__ == '__main__':
    main()
