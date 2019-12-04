from chap07_my_im2col import im2col,col2im
import numpy as np

class Convolution:
    def __init__(self,W,b,stride=1,pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
    def forward(self,x):
        FN,C,FH,FW = self.W.shape
        N,C,H,W = x.shape
        OH = int((H - FH + 2 * self.pad) / self.stride + 1)#手动计算输出大小
        OW = int((W - FW + 2 * self.pad) / self.stride + 1)

        col = im2col(x,FH,FW,self.stride,self.pad)
        w_col = self.W.reshape(FN,-1).T
        out = np.dot(col , w_col) + self.b
        #乘法完成，需要还原，目前是每个输出像素点占一行，每个核对应一列。
        out = out.reshape(N,OH,OW,FN)#折叠4维,比如x (1,3,7,7)通过2,3,5,5卷积，期望得到x (1,2,3,3)，这里临时1,3,3,2，然后变通道
        out = out.transpose(0,3,1,2)#改变通道顺序

        self.col = col
        self.w_col = w_col
        self.x = x

        return out

    def backward(self,dout):
        #因为展开后就和全连接层操作差不多，所以反向传播也差不多，除了同样要处理形状变化，还有col2im的实现（显然，这个应该是最后的操作col变成img嘛）
        #先变形还是后变形？可以这样想，这一层(Conv)就是平铺操作的层，所以所有操作先铺开，所以反向也先变形dout。
        #又因为，模块化的网络，都是全自动的，backward直接就是后一层传入的，没有人替你预处理，所以dout形状肯定是(N,FN,OH,OW)
        FN,C,FH,FW = self.W.shape#
        dout = dout.transpose(0,2,3,1)#先把通道改回来(N,OH,OW,FN)
        dout = dout.reshape(-1,FN)#(N*OH*OW,FN)#这样就和前向传播的乘法结果是一个形状了，可以进行乘法了

        self.dW = np.dot(self.col.T , dout)#注意应该用col，不是x,点积之后
        if 1:#个人认为有两种方法
            self.dW = self.dW.reshape(C,FH,FW,FN)
            self.dW = self.dW.transpose(3,0,1,2)
        else:
            self.dW = self.dW.transpose(1,0)
            self.dW = self.dW.reshape(C,FH,FW,FN)
        dcol = np.dot(dout, self.w_col.T)#dcol只能叫中间变量，最终要dx
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        self.db = np.sum(dout, axis=0)
        return dx


FN = 2
x = np.random.rand(1,3,7,7)
weight = np.random.rand(FN,3,5,5)
bias = np.random.randn(FN,)

net = Convolution(weight,bias,stride=1,pad = 0)
net.forward(x)


