# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

# bring classes directly into package namespace, to save some typing
from similarities.version import __version__
from similarities.text_similarity import BertSimilarity
from similarities.text_similarity import BertSimilarity as Similarity

from similarities.fast_text_similarity import AnnoySimilarity, HnswlibSimilarity
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
    ClipSimilarity,
    SiftSimilarity,
)
from similarities.data_loader import SearchDataLoader
from similarities import evaluation
from similarities import utils
