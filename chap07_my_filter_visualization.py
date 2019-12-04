# coding: utf-8
import sys,os
import numpy as np
import matplotlib.pyplot as plt
print(os.curdir+'/book_dir')
sys.path.append(os.curdir+'/book_dir')
from book_dir.ch07.simple_convnet import SimpleConvNet


def filter_show(filters, nx=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


network = SimpleConvNet()
# 随机进行初始化后的权重
filter_show(network.params['W1'])
# filter_show(network.params['W2'])#想观察一下不同层的，但是这个网络只有一层卷积，其他全连接层是不能打印的
# filter_show(network.params['b1'])

# 学习后的权重
network.load_params("params.pkl")
filter_show(network.params['W1'])
# filter_show(network.params['W2'])