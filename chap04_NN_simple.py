import numpy as np
from chap04_CEE import cross_entropy_error
from chap03_softmax_function import softmax
from chap04_numerical_gradient import numerical_gradient
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)
        # print(self.W.dtype)#做微分，务必确保ndarray的dtype是float
    def predict(self,x):
        return np.dot(x,self.W)
    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)
        return loss

x = np.array([0.6,0.9])
t = np.array([0,0,1])
net = simpleNet()
print(net.W)
print(net.predict(x))
print(np.argmax(net.predict(x)))
print(net.loss(x,t))

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)#传入W，让W去变化，根据固定的x和t，算出微分，靠微分近似得到梯度，不用管公式解析解



