# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

# bring classes directly into package namespace, to save some typing
from similarities.version import __version__
from similarities.bert_similarity import BertSimilarity
from similarities.bert_similarity import BertSimilarity as Similarity

from similarities.fast_bert_similarity import AnnoySimilarity, HnswlibSimilarity
from similarities.literal_similarity import (
    SimHashSimilarity,
    TfidfSimilarity,
    BM25Similarity,
    WordEmbeddingSimilarity,
    CilinSimilarity,
    HownetSimilarity,
    SameCharsSimilarity,
    SequenceMatcherSimilarity,
)
from similarities.image_similarity import (
    ImageHashSimilarity,
    SiftSimilarity,
)
from similarities.clip_similarity import ClipSimilarity
from similarities.clip_module import ClipModule
from similarities.data_loader import SearchDataLoader
from similarities import evaluation
from similarities import utils
from similarities.faiss_bert_similarity import bert_embedding, bert_index, bert_filter, bert_server
from similarities.faiss_clip_similarity import clip_embedding, clip_index, clip_filter, clip_server
from similarities.faiss_bert_similarity import BertClient
from similarities.faiss_clip_similarity import ClipClient, ClipItem
