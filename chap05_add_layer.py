import numpy as np
from chap05_multi_layer import MultiLayer
class AddLayer():
    def __init__(self):
        pass
    def forward(self,x,y):#there is no need to store x and y
        # self.x = x
        # self.y = y
        return x + y

    def backward(self,dout):#dout is upstream
        dx = dout * 1.0
        dy = dout * 1.0
        return dx,dy

# x = 5.0
# y = 1.0
# layer = AddLayer()
# res = layer.forward(x,y)
# print(res)
# dout = 1.0
# dx,dy = layer.backward(dout)
# print(dx,dy)




