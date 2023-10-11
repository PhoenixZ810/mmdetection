# 返回输出结果
import random

import cv2
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt


def feature_vision(f, name, num=25):
    #定义函数，随机从0-end的一个序列中抽取size个不同的数
    def random_num(size, end):
        range_ls = [i for i in range(end)]
        num_ls = []
        for i in range(size):
            num = random.choice(range_ls)
            range_ls.remove(num)
            num_ls.append(num)
        return num_ls

    # f.requires_grad_(False)
    v = f.squeeze(0).cpu().numpy()

    print(v.shape)  # torch.Size([512, 28, 28])

    #随机选取25个通道的特征图
    channel_num = random_num(num, v.shape[0])
    plt.figure(figsize=(10, 10))
    for index, channel in enumerate(channel_num):
        if num == 25:
            ax = plt.subplot(
                5,
                5,
                index + 1,
            )
        else:
            ax = plt.subplot(
                2,
                2,
                index + 1,
            )
        plt.imshow(v[channel, :, :])
    plt.savefig(name + '.jpg', dpi=300)
