# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

This package contains implementations of pairwise similarity queries.
"""

# bring classes directly into package namespace, to save some typing
from similarities.version import __version__
from similarities.similarity import Similarity
from similarities.similarity import (
    cos_sim,
    dot_score,
    semantic_search,
    community_detection,
    pairwise_dot_score,
    pairwise_cos_sim
)

from similarities.fastsim import AnnoySimilarity, HnswlibSimilarity
from similarities.literalsim import (
    SimhashSimilarity,
    TfidfSimilarity,
    BM25Similarity,
    WordEmbeddingSimilarity,
    CilinSimilarity,
    HownetSimilarity
)
