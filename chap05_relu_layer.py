import numpy as np

class ReluLayer():#vectorization
    def __init__(self):
        self.mask = None
    def forward(self,x):
        self.mask = x <= 0
        x2 = x.copy()
        x2[self.mask] = 0#不可能直接传一个x[mask]出去，shape都变了,要复制一份，并且给元素赋值0
        return x2
    def backward(self,dout):
        dout[self.mask] = 0
        return dout


if __name__ == '__main__':
    x = np.arange(-1,3).reshape(2,2)
    print(x)
    layer = ReluLayer()
    y = layer.forward(x)
    print(y)
    dout = np.array([[1.0,1.0],[1.0,1.0]])
    dx = layer.backward(dout)
    print(dx)

    mask_test = x <= 0
    print('mask:',mask_test)
    print('x[mask]:',x[mask_test])
    x[mask_test] = 0
    print('x[mask] = 0:\n',x)
