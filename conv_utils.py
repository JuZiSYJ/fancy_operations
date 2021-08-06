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

class DCT(nn.Module):
    def __init__(self, N = 8, in_channal = 3):
        super(DCT, self).__init__()

        self.N = N  # default is 8 for JPEG
        self.fre_len = N * N
        self.in_channal = in_channal
        self.out_channal =  N * N * in_channal
        # self.weight = torch.from_numpy(self.mk_coff(N = N)).float().unsqueeze(1)


        # 3 H W -> N*N  H/N  W/N
        self.dct_conv = nn.Conv2d(self.in_channal, self.out_channal, N, N, bias=False, groups=self.in_channal)

        # 64 *1 * 8 * 8, from low frequency to high fre
        self.weight = torch.from_numpy(self.mk_coff(N = N)).float().unsqueeze(1)
        # self.dct_conv = nn.Conv2d(1, self.out_channal, N, N, bias=False)
        self.dct_conv.weight.data = torch.cat([self.weight]*self.in_channal, dim=0) # 64 1 8 8
        self.dct_conv.weight.requires_grad = False



        # self.reDCT = nn.ConvTranspose2d(self.out_channal, 1, self.N,  self.N, bias = False)
        # self.reDCT.weight.data = self.weight






    def forward(self, x):
        # jpg = (jpg * self.std) + self.mean # 0-1
        '''
        x:  B C H W, 0-1. RGB
        YCbCr:  b c h w, YCBCR
        DCT: B C*64 H//8 W//8 ,   Y_L..Y_H  Cb_L...Cb_H   Cr_l...Cr_H
        '''
        dct = self.dct_conv(x)
        return dct

    def mk_coff(self, N = 8, rearrange = True):
        dct_weight = np.zeros((N*N, N, N))
        for k in range(N*N):
            u = k // N
            v = k % N
            for i in range(N):
                for j in range(N):
                    tmp1 = self.get_1d(i, u, N=N)
                    tmp2 = self.get_1d(j, v, N=N)
                    tmp = tmp1 * tmp2
                    tmp = tmp * self.get_c(u, N=N) * self.get_c(v, N=N)

                    dct_weight[k, i, j] += tmp
        if rearrange:
            out_weight = self.get_order(dct_weight, N = N)  # from low frequency to high frequency
        return out_weight # (N*N) * N * N

    def get_1d(self, ij, uv, N=8):
        result = math.cos(math.pi * uv * (ij + 0.5) / N)
        return result

    def get_c(self, u, N=8):
        if u == 0:
            return math.sqrt(1 / N)
        else:
            return math.sqrt(2 / N)

    def get_order(self, src_weight, N = 8):
        array_size = N * N
        # order_index = np.zeros((N, N))
        i = 0
        j = 0
        rearrange_weigth = src_weight.copy() # (N*N) * N * N
        for k in range(array_size - 1):
            if (i == 0 or i == N-1) and  j % 2 == 0:
                j += 1
            elif (j == 0 or j == N-1) and i % 2 == 1:
                i += 1
            elif (i + j) % 2 == 1:
                i += 1
                j -= 1
            elif (i + j) % 2 == 0:
                i -= 1
                j += 1
            index = i * N + j
            rearrange_weigth[k+1, ...] = src_weight[index, ...]
        return rearrange_weigth

class ReDCT(nn.Module):
    def __init__(self, N = 4, in_channal = 3):
        super(ReDCT, self).__init__()

        self.N = N  # default is 8 for JPEG
        self.in_channal = in_channal * N * N
        self.out_channal = in_channal
        self.fre_len = N * N

        self.weight = torch.from_numpy(self.mk_coff(N=N)).float().unsqueeze(1)


        self.reDCT = nn.ConvTranspose2d(self.in_channal, self.out_channal, self.N,  self.N, bias = False, groups=self.out_channal)
        self.reDCT.weight.data = torch.cat([self.weight]*self.out_channal, dim=0)
        self.reDCT.weight.requires_grad = False


    def forward(self, dct):
        '''
        IDCT  from DCT domain to pixle domain
        B C*64 H//8 W//8   ->   B C H W
        '''
        out = self.reDCT(dct)
        return out

    def mk_coff(self, N = 8, rearrange = True):
        dct_weight = np.zeros((N*N, N, N))
        for k in range(N*N):
            u = k // N
            v = k % N
            for i in range(N):
                for j in range(N):
                    tmp1 = self.get_1d(i, u, N=N)
                    tmp2 = self.get_1d(j, v, N=N)
                    tmp = tmp1 * tmp2
                    tmp = tmp * self.get_c(u, N=N) * self.get_c(v, N=N)

                    dct_weight[k, i, j] += tmp
        if rearrange:
            out_weight = self.get_order(dct_weight, N = N)  # from low frequency to high frequency
        return out_weight # (N*N) * N * N

    def get_1d(self, ij, uv, N=8):
        result = math.cos(math.pi * uv * (ij + 0.5) / N)
        return result

    def get_c(self, u, N=8):
        if u == 0:
            return math.sqrt(1 / N)
        else:
            return math.sqrt(2 / N)

    def get_order(self, src_weight, N = 8):
        array_size = N * N
        # order_index = np.zeros((N, N))
        i = 0
        j = 0
        rearrange_weigth = src_weight.copy() # (N*N) * N * N
        for k in range(array_size - 1):
            if (i == 0 or i == N-1) and  j % 2 == 0:
                j += 1
            elif (j == 0 or j == N-1) and i % 2 == 1:
                i += 1
            elif (i + j) % 2 == 1:
                i += 1
                j -= 1
            elif (i + j) % 2 == 0:
                i -= 1
                j += 1
            index = i * N + j
            rearrange_weigth[k+1, ...] = src_weight[index, ...]
        return rearrange_weigth

    def conv_merge():
    '''
    1*1  +  3*3  ->  3*3,  no bias
    :return:
    '''
    x = torch.rand(5,3,112,112)
    conv_1 = nn.Conv2d(3,2,1,1,0, bias= False) # c d 1 1
    conv_3 = nn.Conv2d(2,4,3,1,1, bias= False) # d e 3 3

    new_conv_1 = nn.Conv2d(2,3,1,1, bias=False)
    new_conv_1.weight.data = conv_1.weight.data.permute(1,0,2,3) # d c 1 1

    conv_merge = nn.Conv2d(3,4,3,1,1,bias=False)
    merge_weight = new_conv_1(conv_3.weight) # d c 3 3
    conv_merge.weight.data = merge_weight

    y1 = conv_3(conv_1(x))
    y2 = conv_merge(x)
    print(torch.sum(torch.abs(y1 - y2))) # 0.0036

def conv_merge2():
    '''
    3*3 + 1*1  -> 3*3
    :return: 
    '''
    ori_3_3 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride = 1, padding = 1, bias=True)
    ori_1_1 = nn.Conv2d(in_channels=10, out_channels=6, kernel_size= 1, stride=1, padding=0, bias=True)

    weight_3_3_ = ori_3_3.weight.data.clone()
    weight_1_1_ = ori_1_1.weight.data.clone()

    bias_3_3_ = ori_3_3.bias.data.clone()
    bias_1_1 = ori_1_1.bias.data.clone()


    reweight_3_3 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1, bias=True)
    print(reweight_3_3.weight.data.shape)

    for i in range(weight_1_1_.shape[0]):
        reweight_3_3.weight.data[i,...] = torch.sum(weight_3_3_ * weight_1_1_[i,...].unsqueeze(1), dim=0)

        reweight_3_3.bias.data[i] = bias_1_1[i] + \
                                    torch.sum(bias_3_3_ * weight_1_1_[i,...].squeeze(1).squeeze(1))

    model = nn.Sequential(ori_3_3, ori_1_1)

    x = torch.tensor(np.array(Image.open('1.jpeg'))).unsqueeze(0).permute(0,3,1,2).float()

    out = model(x)
    out2 = reweight_3_3(x)

    print(torch.sum(torch.abs(out - out2)), out, out2) # tensor(5.4640, grad_fn=<SumBackward0>)


if __name__ == '__main__':
    pass
