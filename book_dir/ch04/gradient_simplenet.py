# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t)# 固定住x和t，通过f调用loss并通过loss调用predict间接使用W，
# 关键是，怎么确定w是f的输入的？其实这里也只是f(lambda)函数，后边传入了net.W
dW = numerical_gradient(f, net.W)##怎么确定的W和x、t的关系？因为net对象被传入了lambda，怎么确定要计算w的导数？在外部调用lambda时再传入net对象的W。

print(dW)
