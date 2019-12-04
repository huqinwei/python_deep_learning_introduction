import sys,os
# from book_dir.common.util import im2col
from chap07_my_im2col import im2col,im2col2
import numpy as np
import time

test_time_consumed = False
if test_time_consumed:

    x1 = np.random.rand(100000,3,7,7)

    start1 = time.time()
    col1 = im2col(x1,5,5,1,pad=0)
    end1 = time.time()

    start2 = time.time()
    col1_2 = im2col2(x1,5,5,1,pad=0)
    end2 = time.time()

    print(np.average(col1==col1_2))

    print(col1.shape,col1_2.shape)
    print(end1-start1,'\n',end2-start2)

test_pad = True
if test_pad:#
    x1 = np.random.rand(1,3,9,9)
    col0 = im2col(x1,3,3,1,pad=0)
    x1 = np.random.rand(10,3,9,9)
    col1 = im2col(x1,3,3,1,pad=0)
    col2 = im2col(x1,3,3,1,pad=1)
    col3 = im2col(x1,5,5,1,pad=1)
    print(col0.shape)
    print(col1.shape)
    print(col2.shape)#shape的后边很好计算，就是窗口size乘以channel数。
    print(col3.shape)
