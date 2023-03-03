#!/usr/bin/env python
# _*_ coding:utf-8 _*_

# @Time: 2022/7/5 22:08
# @author: xiao chen
# @File: PFA_main.py

import torch
from .Algorithm_4 import Algorithm_4
 
def PFA_main(mean1, mean2, mean3, threshold:float, device):
      """
      inputs:
         mean1: tensor---batch_size x feature_num   dna.
         mean2: tensor---batch_size x feature_num   rna.
         mean3: tensor---batch_size x feature_num   rppa.
         threshold: the arrange of value is from 0 to 1. 
         device: put data into CPU or GPU.
      outputs: 
         return: fusion data, used for adjacency matrix--- batch_size x batch_size
      """
      temp1 = mean1.T.detach().cpu().numpy()
      temp2 = mean2.T.detach().cpu().numpy()
      temp3 = mean3.T.detach().cpu().numpy() 
      XList = []
      XList.append(temp1)
      XList.append(temp2)
      XList.append(temp3)
      d_num = min(mean1.shape[1], mean2.shape[1], mean3.shape[1])
      # print(d_num)

      Y, w, L_list = Algorithm_4(XList, sample_num=mean1.shape[0], iter_num=1000, lam_1=1, d_num=mean1.shape[0], k=3)
      # Y, w, L_list = Algorithm_4(XList, sample_num=mean1.shape[0], iter_num=1000, lam_1=1, d_num=d_num, k=3)
      result = torch.tensor(Y.T)
      result = result.to(device)
      result[result > threshold] = 1
      result[result <= threshold] = 0

      return result

 