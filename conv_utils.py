# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :fancy_operations
# @File     :conv_utils.py
# @Date     :2020/12/30 下午8:02
# @Author   :SYJ
# @Email    :JuZiSYJ@gmail.com
# @Software :PyCharm
-------------------------------------------------
"""
import cv2
import numpy as np
from PIL import Image
import math
import torch
import os
from io import BytesIO
from torch import nn
import torch.nn.functional as F


class MeanShift(nn.Conv2d):
    '''
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        x = (x - mean) / std
    '''
    def __init__(
        self, rgb_range=1.0,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

def conv_implement_by_unfold():
    '''
    https://www.freesion.com/article/63531128664/
    :return:
    '''
    inp = torch.randn(2, 3, 4, 5)
    w = torch.randn(2, 3, 3, 3)
    inp_unf = torch.nn.functional.unfold(inp, (3, 3))  # [2,27,6]
    out_unf = inp_unf.transpose(1, 2).matmul(
        w.view(w.size(0), -1).t()).transpose(1, 2)

class dct(nn.Module):
    def __init__(self):
        super(dct, self).__init__()


        self.dct_conv = nn.Conv2d(3,192,8,8,bias=False, groups=3) # 3 h w -> 192 h/8 w/8
        self.weight = torch.from_numpy(np.load('models/DCTmtx.npy')).float().permute(2,0,1).unsqueeze(1)# 64 1 8 8, order in Z
        self.dct_conv.weight.data  =  torch.cat([self.weight] * 3, dim=0) # 192 1 8 8
        self.dct_conv.weight.requires_grad  = False

        self.mean = torch.Tensor([[[[0.485, 0.456, 0.406]]]]).reshape(1, 3, 1,
                                                                 1)
        self.std = torch.Tensor([[[[0.229, 0.224, 0.225]]]]).reshape(1, 3, 1,
                                                                1)
        self.Ycbcr = nn.Conv2d(3, 3, 1, 1, bias=False)
        trans_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.5],
                                 [0.5, -0.419, -0.081]])
        trans_matrix = torch.from_numpy(trans_matrix).float().unsqueeze(
            2).unsqueeze(3)
        self.Ycbcr.weight.data = trans_matrix
        self.Ycbcr.weight.requires_grad = False

        self.reYcbcr = nn.Conv2d(3, 3, 1, 1, bias=False)
        re_matrix = np.linalg.pinv(np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.5],
                                 [0.5, -0.419, -0.081]]))
        re_matrix = torch.from_numpy(re_matrix).float().unsqueeze(
            2).unsqueeze(3)
        self.reYcbcr.weight.data = re_matrix

    def forward(self, x):

        # jpg = (jpg * self.std) + self.mean # 0-1
        ycbcr = self.Ycbcr(x) # b 3 h w

        dct = self.dct_conv(ycbcr)
        return dct

    def reverse(self,x):
        dct = F.conv_transpose2d(x, torch.cat([self.weight] * 3,0), bias=None, stride=8, groups = 3)
        rgb = self.reYcbcr(dct)
        return rgb


# 暗通道先验计算单元
def minpool(feat, ksize, stride=1):
    pad = (ksize - 1)// 2
    N, C, H, W = feat.size()
    feat = F.pad(feat, (pad, pad, pad, pad),mode='reflect')
    feat = feat.unfold(2,ksize,stride).unfold(3,ksize,stride)
    feat = feat.permute(0,2,3,1,4,5).contiguous()
    feat = feat.view(N,H,W,-1)
    out  = feat.min(-1)[0]
    out  = out.unsqueeze(1)
    return out



if __name__ == '__main__':
    pass
