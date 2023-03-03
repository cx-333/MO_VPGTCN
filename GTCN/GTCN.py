#!/usr/bin/env python
# _*_ coding:utf-8 _*_

# @Time: 2022/7/5 22:38
# @author: xiao chen
# @File: GTCN.py

import sys, os
print(os.getcwd())
sys.path.append("data_fusion/PFA/")
from Algorithm_4 import Algorithm_4
import numpy as np
 
k = 3
lam_1 = 1
iter_num = 1000
sample_num = 36
xList = []
for i in range(k):
   xList.append(0)
xList[0] = np.random.random((32, sample_num))
xList[1] = np.random.random((40, sample_num))
xList[2] = np.random.random((50, sample_num))

Y, w, L_list = Algorithm_4(xList, sample_num, iter_num, lam_1, d_num=36, k=k)

print('#'*200)
print(np.diag(Y.dot(Y.T)))
print('#'*200)
print(np.sum(w >= 0))
print('#'*200)
print(L_list)
print(Y.shape)

