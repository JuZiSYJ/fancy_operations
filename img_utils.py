# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :fancy_operations
# @File     :img_utils.py
# @Date     :2020/12/30 下午7:41
# @Author   :SYJ
# @Email    :JuZiSYJ@gmail.com
# @Software :PyCharm
-------------------------------------------------
"""

'''
some function is borrow from https://github.com/cszn/KAIR
'''
import cv2
import numpy as np
from PIL import Image
import math
import torch
import os
from io import BytesIO
from torch import nn
import torch.nn.functional as F

# convert 2/3/4-dimensional torch tensor to uint
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def from_bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB

def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    numpy image of WxHxC or WxH
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)



# --------------------------------------------
# PSNR
# --------------------------------------------
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def get_angular_loss(vec1,vec2):
    safe_v = 0.999999
    illum_normalized1 = torch.nn.functional.normalize(vec1,dim=1)
    illum_normalized2 = torch.nn.functional.normalize(vec2,dim=1)
    dot = torch.sum(illum_normalized1*illum_normalized2,dim=1)
    dot = torch.clamp(dot, -safe_v, safe_v)
    angle = torch.acos(dot)*(180/math.pi)
    loss = torch.mean(angle)
    return loss

def cos_distance(src, dst):
    src_norm = np.linalg.norm(src, axis=-1, keepdims=True)
    dst_norm = np.linalg.norm(dst, axis=-1, keepdims=True)

    src_norm2 = src / src_norm # have some item is 0, so cause None
    dst_norm2 = dst / dst_norm

    distance = np.sum(src_norm2 * dst_norm2, axis=-1)
    distance = np.clip(distance, -1, 1)

    angle = np.arccos(distance) * 180 / np.pi

    angle_95 = np.nanpercentile(angle, 95)
    angle_mean = np.nanmean(angle)

    return angle_95, angle_mean, angle

def produce_jpeg():
    img = Image.open('..')
    buffer = BytesIO()
    img.save(buffer, 'JPEG', quality=50)
    out_path = ''
    new = Image.open(buffer)
    new.save(out_path)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class LPIPS(torch.nn.Module):
    """Learned Perceptual Image Patch Similarity.
    Args:
        if_spatial: return a score or a map of scores.
        im: cv2 loaded images, or ([RGB] H W), [0, 1] CUDA tensor.
    https://github.com/richzhang/PerceptualSimilarity

    ref = cv2.imread('test.bmp')
    # 0.012859745882451534  for [-10,10],
    # #0.001899  for [-5,5], less is better
    im = cv2.imread('test.bmp') + np.random.randint(-5,5,size=ref.shape)
    lpips = LPIPS()
    score1  = lpips(ref, im)
    print(score1)

    """
    def __init__(self, net='alex', if_spatial=False, if_cuda=False):
        super().__init__()
        import lpips

        self.lpips_fn = lpips.LPIPS(net=net, spatial=if_spatial)
        if if_cuda:
            self.lpips_fn.cuda()

    def _preprocess(self, inp, mode):
        if mode == 'im':
            im = inp[:, :, ::-1]  # (H W BGR) -> (H W RGB)
            im = im / (255. / 2.)  - 1.
            im = im[..., np.newaxis]  # (H W RGB 1)
            im = im.transpose(3, 2, 0, 1)  # (B=1 C=RGB H W)
            out = torch.Tensor(im)
        elif mode == 'tensor':
            out = inp * 2. - 1.
        return out

    def forward(self, ref, im):
        mode = 'im' if ref.dtype == np.uint8 else 'tensor'
        ref = self._preprocess(ref, mode=mode)
        im = self._preprocess(im, mode=mode)
        lpips_score = self.lpips_fn.forward(ref, im)
        return lpips_score.item()

if __name__ == '__main__':
    pass