from chap04_numerical_diff import func3,numerical_diff

import numpy as np

def func_3d(x):
    return x[0] ** 2 + x[1] ** 2
def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)
def function_x0(x):#x1=4 given
    return x ** 2 + 4 ** 2
def function_x1(x):#x0=3 given
    return 3 ** 2 + x ** 2

def numerical_gradient_mine(f,x):#func_3d
    grad = np.zeros_like(x)#dtype=np.float
    # print(grad.dtype)
    h = 1e-4
    for idx in range(len(x)):
        tmp = x[idx]
        x[idx] = float(tmp) + h
        fxh1 = f(x)
        x[idx] = tmp - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp
    return grad

def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = float(tmp_val) - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):#super slow~!!!!!!!!!!!!!!!!!!!!!!!!!!
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad

import matplotlib.pyplot as plt
if __name__ == '__main__':
    x = [np.arange(0.0,20.0,0.1),np.arange(5.0,25.0,0.1),]
    y = func_3d(x)
    # plt.plot(x,y)#this is 2d line
    x0 = 3
    x1 = 4
    print(numerical_diff(function_x0,x0))
    print(numerical_diff(function_x1,x1))


    x = [x0,x1]
    x = [float(x0),float(x1)]# 踩坑 ref:chap04_ndarray_list_dtype_change.py
    grad = numerical_gradient(func_3d,np.array(x))#cant change ndarray.dtype dynamically!!!!!!!!!!!!!!!!!!list can
    print(grad.dtype)
    print('grad is ',grad)
    # grad2 = numerical_gradient_mine(func_3d,np.array(x))#the same as numerical_gradient_no_batch
    # print(grad2.dtype)
    # print('grad2 is ',grad2)
    print(numerical_gradient(func_3d,np.array([3.0,4.0])))
    print(numerical_gradient(func_3d,np.array([0.0,2.0])))
    tmp_to_print = numerical_gradient_mine(func_3d,np.array([3.0,0.0]))
    print(tmp_to_print)#被改成易读形式，小数点后看不到了，numpy干的


    x0 = np.arange(0.0,20.0,0.1)
    x1 = np.arange(0.0,20.0,0.1)
    x0 = np.arange(0.0,3.0,0.1)
    x1 = np.arange(0.0,3.0,0.1)
    x0 = np.arange(-2, 2.5, 0.25)#圆心对称，所以用那个有正负的好
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)#理解错了，这已经产出Y了，后边不是自己计算Y了，这个Y其实是坐标轴（x1）
    print(X.shape,Y.shape)
    X = X.flatten()
    Y = Y.flatten()
    print(X.shape,Y.shape)
    grad = numerical_gradient(function_2,np.array([X,Y]))#分别得到X轴和Y轴的微分，其实分开写再concatenate应该差不多,后边还拆开呢，主要是结合用法
    print(Y.shape)
    # Y = Y.sum(axis=0)#随便做一个无意义的操作，只为打印
    #负梯度是梯度下降的方向，指向最低点，当然，如果是复杂图形，这么说就不严谨了，这个算凸函数？

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")  # ,headwidth=10,scale=40,color="#444444")#先不研究画箭头了#https://blog.csdn.net/liuchengzimozigreat/article/details/84566650
    # plt.plot(X,Y)
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()



#默认float64，却可以显示不带小数点的6 7
#为什么函数内的zeros_like是int32，这是float64,因为x在函数内部已经被修改过，x内部的值变了类型
#问题出在函数内部忘了用tmp给x重新赋值，修复那个bug，这个现象也不在了
# print('changed x:',type(x[0]))
# x = [x0,x1]#redefine x
# print('origin x',type(x[0]))
# grad_new = np.zeros_like(x)
# print(grad_new.dtype)
# print(grad_new)
# grad_new[0] = 6.1
# grad_new[1] = 7.0
# print(grad_new)
# print(grad_new.dtype)





