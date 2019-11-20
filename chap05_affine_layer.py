import numpy as np


class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None


    def forward(self,x):
        self.x = x
        out = np.dot(x,self.W) + self.b
        return out

    def backward(self,dout):
        self.db = np.sum(dout,axis = 0)
        self.dW = np.dot(self.x.T,dout)
        dx = np.dot(dout,self.W.T)
        return dx

if 0:
    class Affine:
        def __init__(self, W, b):
            self.W = W
            self.b = b

            self.x = None
            self.original_x_shape = None
            # 权重和偏置参数的导数
            self.dW = None
            self.db = None

        def forward(self, x):
            # 对应张量
            self.original_x_shape = x.shape  # todo 这有什么用？
            x = x.reshape(x.shape[0], -1)
            self.x = x

            out = np.dot(self.x, self.W) + self.b

            return out

        def backward(self, dout):
            dx = np.dot(dout, self.W.T)
            self.dW = np.dot(self.x.T, dout)
            self.db = np.sum(dout, axis=0)

            dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
            return dx


if __name__ == '__main__':
    X = np.random.randn(2)
    W = np.random.randn(2,3)
    b = np.random.randn(3)
    y = np.dot(X,W) + b
    print('shape of y:',y.shape)

    k = 10
    X = np.random.randn(k,2)
    W = np.random.randn(2,3)
    b = np.random.randn(3)
    y = np.dot(X,W) + b#affine layer
    print('shape of y:',y.shape)

    #broadcast of bias   in  batch_size = 2
    X_dot_W = np.array([[0,0,0],[10,10,10]])
    B = np.array([1,2,3])
    print('X_dot_W + B:',X_dot_W + B)

    dY = np.array([[1,2,3],[4,5,6]])
    print(dY)
    dB = np.sum(dY,axis = 0)
    print(dB)

    k = 5
    x = np.random.randn(k,2)
    W = np.random.randn(2,3)
    b = np.random.randn(3)
    aff = Affine(W,b)
    y = aff.forward(x)
    print('y:',y)
    dy = np.ones_like(y)
    print('dy.shape:',dy.shape)
    dx = aff.backward(dy)
    print(dx.shape)
    print(dx,aff.dW,aff.db)
