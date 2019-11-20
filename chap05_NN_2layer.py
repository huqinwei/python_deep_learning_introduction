#相比chap04的NN，多了一些梯度计算？用上了层接口，都是这章定义的
#把所有需要的参数罗列出来，所有需要的接口导入，顺便复习一下
#因为他没明确给出实际使用，所以有两个grad相关接口不知道都干什么用的
#区别1，chap04的参数在init，层的搭建在predict，现在层也在init完成?并不是，是层的定义在init！！搭建还得是predict
#那些dot操作都封装了，用affine层:affine1->relu1->affine2->softmax
#predict的接口不走softmax，他是把softmax独立作为final layer了，而不是网络结构layers。省一步，不经过softmax，直接用argmax（但是要在predict()外部用）

import numpy as np

from chap05_affine_layer import Affine
from chap05_relu_layer import ReluLayer
from chap05_softmax_layer import SoftmaxWithLoss

from collections import OrderedDict
from chap04_numerical_gradient import numerical_gradient



#todo 他这里很多东西没明确怎么用，所以实现也就有浮动
class TwoLayerNet():#括号为了继承，如果不继承，写不写括号一样
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size,hidden_size) * weight_init_std
        # self.params['b1'] = np.random.randn(hidden_size)# * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size,output_size) * weight_init_std#todo 这乘不乘，差很多
        # self.params['b2'] = np.random.randn(output_size)# * weight_init_std
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['relu1'] = ReluLayer()
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['b2'])

        self.final_layer = SoftmaxWithLoss()
    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    def loss(self,x,t):#one-hot
        y = self.predict(x)
        loss = self.final_layer.forward(y,t)
        return loss

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:#one-hot to num
            t = np.argmax(t,axis=1)
        acc = np.sum(y==t) / y.shape[0]#there s no need to cast to float    float(y.shape[0])
        return acc

    def numerical_gradient(self,x,t):#为什么把两种都列出来了？他也没说使用场景，也许是“为了用微分解来验证解析解”
        grads = {}
        # f = lambda x:self.loss(x,t)#这样写，求的是对x的导数啊
        loss_W = lambda W:self.loss(x,t)#注意，这个W是参数，而loss传进去的是x、t，W作为loss_W的输入，是隐含的
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])
        return grads
    def gradient(self,x,t):#backpropagation
        grads = {}

        # 反向传播的起点不是loss？仍然用1.0？他是想最后乘？
        # #而微分解怎么就用上了loss？必须得用吧，他改变任何一点，都是观察loss的变化从而得出微分
        # 微分本身是相对于这个点x，反向传播是根据公式，不需要loss，1.0就是这么一个起点
        loss = self.loss(x,t) # 也许这里是省略了，看最后怎么用吧
        #或者loss只是loss了，和求梯度无关（）？那backward还有/batch_size干什么？
        # 如果dout是1.0，什么含义？就是loss每变1.0，前边的参数应该对应的导数？
        # 如果单从量级考虑，每次都固定传10，那么只能用lr去控制step了，而不能根据loss大小自动变化
        # 因为微分法是纯粹的试出来的梯度，所以根本不需要有反向传播这一概念，反向传播是解析法专有
        # 所以，这个书大量借鉴CS231N，这里把两种都写出来，就是说用微分法验证解析法的正确性


        dout = 1.0
        dout = self.final_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()#reverse in place

        for layer in layers:#todo 这里写顺手了，后边提取W、b确实用了self.layers[key]，但是这里明明是reverse得到的layers，不是self.layers，
            # 那能不能，从layers顺便也得到W和b呢？能得到，但是没有逻辑关系，因为是list，不是key，并且麻烦的一点是，这还没BP完呢，你必须BP一步提取一步。
            dout = layer.backward(dout)#反向传播这个过程走过去，每个参数的梯度就有了（dW、db），不需要用这个dout，直接用dW、db
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

if __name__ == '__main__':

    net = TwoLayerNet(2,10,2)
    x = np.array([[1.0,2.0],[11.0,2.0]])
    t = np.array([[0,1],[0,1]])
    y = net.predict(x)
    print(y)
    print(y.argmax(axis=1))

    print(5 / 10)

    print(net.layers.values())
    layers = list(net.layers.values())
    print(layers)
    print(layers.reverse())
    print(layers)


#todo verify numerical grad and analytic grad

    api_test = True
    if api_test:
        print('hello')
        #E:\MachineLearning\ML_WIN_PROJECTS\python_DL_from_scratch\scratch1\chap05_OrderedDict.py







