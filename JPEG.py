# -*- coding: utf-8 -*-
# @Time    : 2021/3/4 
# @Author  : SYJ
# @Email   : juzisyj@gmail.com
# @File    : test.py
# @Software: PyCharm

import os
import torch
import numpy as np
from io import BytesIO
from PIL import Image as im
from PIL import JpegPresets
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr
from PIL import Image
from matplotlib import pyplot as plt


qf_table = [[[16, 11, 10, 16, 24, 40, 51, 61],
             [12, 12, 14, 19, 26, 58, 60, 55],
             [14, 13, 16, 24, 40, 57, 69, 56],
             [14, 17, 22, 29, 51, 87, 80, 62],
             [18, 22, 37, 56, 68, 109, 103, 77],
             [24, 35, 55, 64, 81, 104, 113, 92],
             [49, 64, 78, 87, 103, 121, 120, 101],
             [72, 92, 95, 98, 112, 100, 103, 99]],

            [[17, 18, 24, 47, 99, 99, 99, 99],
             [18, 21, 26, 66, 99, 99, 99, 99],
             [24, 26, 56, 99, 99, 99, 99, 99],
             [47, 66, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99]]
            ]
qf = 10

def new_qf(qf_table, qf):
    if qf < 50:
        m = 5000 / qf
    else:
        m = 200 - 2 * qf
    qf_table = np.array(qf_table).reshape(2,64) # 2*8*8


    qf_table = np.clip(np.floor((qf_table * m + 50) / 100 ), 1, a_max=255)
    qf_table = qf_table.astype('uint8').tolist()

    return qf_table

new_qftable = new_qf(qf_table, qf)

JpegPresets.presets['new_qf'] = {
    'quantization': new_qftable,
    'subsampling': 2 ##
}


def test(tmp):

    # tmp = im.fromarray(image)
    CompressBuffer.seek(0)
    # tmp.save(CompressBuffer, "JPEG", quality='matlab10')
    tmp.save(CompressBuffer, "JPEG", quality='new_qf')
    CompressBuffer.seek(0)
    tmp = np.asarray(im.open(CompressBuffer))
    return tmp


if __name__ == '__main__':
    CompressBuffer = BytesIO()

    img = Image.open('test.bmp')
    compress = test(img)
    plt.imshow(compress)



    CompressBuffer.seek(0)
    img.save(CompressBuffer, "JPEG", quality=10)
    CompressBuffer.seek(0)

    tmp2 = np.asarray(im.open(CompressBuffer))

    print("sub sum:{}".format(np.sum(np.abs(tmp2 * 1.0 - compress * 1.0)))) # sub sum:0.0

    plt.subplot(1, 3, 1)
    plt.imshow(compress)

    plt.subplot(1, 3, 2)
    plt.imshow(tmp2)

    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(tmp2 * 1.0 - compress * 1.0).astype('uint8'))

    plt.show()