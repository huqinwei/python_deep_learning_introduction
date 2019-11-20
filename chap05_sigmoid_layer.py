import numpy as np

class SigmoidLayer():
    def __init__(self):
        self.out = None
    def forward(self,x):
        self.out = 1 / ( 1 + np.exp(-x))
        return self.out
    def backward_wrong(self,dout):#错在哪了？导数是y*(1-y)，但是传进来的不是y啊，是DL/dy，其实是DL/dy * y * (1-y)，所以还是有必要保存之前的x或者y的
        d = dout * (1 - dout)
        return d
    def backward(self,dout):
        dx = dout * self.out * (1 - self.out)
        return dx


if __name__ == '__main__':
    layer = SigmoidLayer()
    x = 1
    y = layer.forward(x)
    dy = 1.0
    dx = layer.backward(dy)
    print(x,y)
    print(dx,dy)

