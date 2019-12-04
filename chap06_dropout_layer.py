import numpy as np

class Dropout:
    def __init__(self,dropout_ratio=0.5):#这里的是扔的概率
        self.dropout_ratio = dropout_ratio
        self.mask = None
    def forward(self,x,is_train):
        if is_train:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1 - self.dropout_ratio)
    def backward(self,dout):
        return dout * self.mask



class Dropout:
    def __init__(self,keep_probability=0.5):#这里的是保留的概率
        self.keep_probability = keep_probability
        self.mask = None
    def forward(self,x,is_train):
        if is_train:
            self.mask = np.random.rand(*x.shape) < self.keep_probability
            return x * self.mask
        else:
            return x * self.keep_probability
    def backward(self,dout):
        return dout * self.mask

class Dropout:
    def __init__(self,keep_probability=0.5):#这里的是保留的概率
        self.keep_probability = keep_probability
        self.mask = None
    def forward(self,x,is_train=True):
        if is_train:
            self.mask = np.random.rand(*x.shape) < self.keep_probability
            return x * self.mask / self.keep_probability
        else:
            return x
    def backward(self,dout):
        return dout * self.mask








