import numpy as np
#non-vectorization:这里换成多维数组都是错的，这是数字积，不是点积
class MultiLayer():
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self,x,y):
        self.x = x
        self.y = y
        return self.x * self.y

    def backward(self,dout):#dout is upstream
        dx = dout * self.y
        dy = dout * self.x
        return dx,dy

# x = 5.0
# y = 1.0
# layer = MultiLayer()
# res = layer.forward(x,y)
# dout = 1.0
# dx,dy = layer.backward(dout)
# print(dx,dy)

if __name__ == '__main__':
    apple_price = 100
    apple_num = 2
    tax_rate = 1.1
    layer1 = MultiLayer()
    layer2 = MultiLayer()
    res1 = layer1.forward(apple_price,apple_num)
    price = layer2.forward(res1,tax_rate)
    print('final price:',price)
    dprice = 1.0
    dres1,dtax_rate = layer2.backward(dprice)#对应关系，同一层，forward的输出price对应backward的输入dprice
    dapple_price,dapple_num = layer1.backward(dres1)
    print(dres1,dtax_rate)
    print(dapple_price,dapple_num)


    #non-vectorization
    vector_a = np.array([[1.0,2.0],[3.0,4.0]])
    vector_b = np.array([[2.0,2.0],[3.0,4.0]])
    test_layer = MultiLayer()
    res = test_layer.forward(vector_a,vector_b)
    print('res is ',res)
    res = np.dot(vector_a,vector_b)
    print('res is ',res)


