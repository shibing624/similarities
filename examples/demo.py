# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import numpy as np
import gensim
from gensim.matutils import *
from gensim import matutils
from scipy.sparse import csc_matrix, csr_matrix

vec_1 = [(2, 1), (3, 4), (4, 1), (5, 1), (1, 1), (7, 2)]
vec_2 = [(1, 1), (3, 8), (4, 1)]
result = matutils.jaccard(vec_2, vec_1)
expected = 1 - 0.3
print(result)

# checking ndarray, csr_matrix as inputs
vec_1 = np.array([[1, 3], [0, 4], [2, 3]])
vec_2 = csr_matrix([[1, 4], [0, 2], [2, 2]])
result = matutils.jaccard(vec_1, vec_2)
expected = 1 - 0.388888888889
print(result)

# checking ndarray, list as inputs
vec_1 = np.array([6, 1, 2, 3])
vec_2 = [4, 3, 2, 5]
result = matutils.jaccard(vec_1, vec_2)
expected = 1 - 0.333333333333
print(result)

vec_1 = [[1, 3], [2, 4], [3, 3]]
vec_2 = [[1, 6], [2, 2], [3, 2]]

vec_1 = [[0, 1], [1, 4], [2, 6]]
vec_2 = [[0, 1], [1, 2], [2, 3]]
a = cossim(vec_1, vec_2)
print(a)

vec_1 = [[0, 1], [1, 1], [2, 1]]
vec_2 = [[0, 1], [1, 2], [2, 3]]
a = cossim(vec_1, vec_2)
print(a)

vec_1 = [[0, 2], [1, 4], [2, 6]]
vec_2 = [[0, 1], [1, 2], [2, 3]]
a = cossim(vec_1, vec_2)
print(a)
print("jaccard:", matutils.jaccard(vec_1, vec_2))

vec_1 = np.array([2,4,6])
vec_2 = np.array([1,2,3])

# vec_1 = np.array([3,4,3])
# vec_2 = np.array([6,2,2])
#
# vec_1 = np.array([[3],[4],[3]])
# vec_2 = np.array([[6],[2],[2]])
print("jaccard2:", matutils.jaccard(vec_1, vec_2))

