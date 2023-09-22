# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: cli entry point
"""
import sys

import fire

sys.path.append('..')
from similarities.faiss_bert_similarity import bert_embedding, bert_index, bert_filter, bert_server
from similarities.faiss_clip_similarity import clip_embedding, clip_index, clip_filter, clip_server


def main():
    """Main entry point"""

    fire.Fire(
        {
            "bert_embedding": bert_embedding,
            "bert_index": bert_index,
            "bert_filter": bert_filter,
            "bert_server": bert_server,
            "clip_embedding": clip_embedding,
            "clip_index": clip_index,
            "clip_filter": clip_filter,
            "clip_server": clip_server,
        }
    )


if __name__ == "__main__":
    main()
