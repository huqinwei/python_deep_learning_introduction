import numpy as np
from chap03_softmax_function import softmax,softmax_batch
from chap04_CEE import cross_entropy_error_one_hot
import time

class SoftmaxWithLoss():
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self,x,t):
        #测了一下两种实现的速度差距，在mnist 3000个数据的情况下，50倍以上
        #softmax no batch: 0.051000118255615234
        #softmax batch:    0.0009999275207519531
        # start = time.time()
        # y1 = softmax(x)
        # print('softmax no batch:',time.time()-start)
        # start = time.time()
        y = softmax_batch(x)
        # print('softmax batch:',time.time()-start)
        # print('does this two y equal?\n',y1 == y)#yes!@@!!!!!!!!!!!!!!!!!!!@!!!!!!!!

        self.y = y
        self.t = t
        self.loss = cross_entropy_error_one_hot(y,t)#is already loss,not just a local result
        return self.loss


    def backward(self,dout = 1.0):
        batch_size = self.y.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


if __name__ == '__main__':
    x = np.array([[6,2,5],[1,10,0]])#注意这个量级，一定要在同一量级？否则概率合不为1？好像不太对啊,因为我的softmax的sum没有按axis
    t = np.array([[0,1,0],[0,1,0]])
    layer = SoftmaxWithLoss()
    loss = layer.forward(x,t)
    print('loss:',loss)
    dx = layer.backward()
    print('dx:',dx)


