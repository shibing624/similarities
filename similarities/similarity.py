# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from typing import List, Union, Optional
import numpy as np
from numpy import ndarray
from torch import Tensor
from loguru import logger


class BertSimilarity:
    def __init__(self, model_name_or_path=''):
        """
        Cal text similarity
        :param similarity_type:
        :param embedding_type:
        """
        self.model_name_or_path = model_name_or_path
        self.model = None

    def encode(self, sentences: Union[List[str], str]) -> ndarray:
        return np.array([])

    def similarity_score(self, sentences1: Union[List[str], str], entences2: Union[List[str], str]):
        """
        Get similarity scores between sentences1 and sentences2
        :param sentences1: list, sentence1 list
        :param sentences2: list, sentence2 list
        :return: return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        return 0.0
