#之前他没提到，但是optimizer他用到了，实现一个多层网络，虽然本质上没有比2层网络高级，但是涉及一些实现的代码细节和循环简化，练一下手
#大体结构和之前的2层网络一致，但是网络层数是灵活的，内部靠一个循环就可以实现。
#照旧，softmax是分离开来的，中间只要循环affine和relu就好了
#todo 警告：原文的参考代码极其混乱，尤其是计数和下标使用混乱，所以闭着眼睛几乎不可能和他写出一样的，只要逻辑保证对就行了。照着他的改太不舒服了，改的我人都消极了。。。。
# 原文多数代码都是一种差不多能用就行，使用方法要照顾内部实现的标准，所以很难揣摩他的内部实现到底搞成什么样，当然不能和标准库去比规范性。
#所有有问题的位置用todo标出，用于在pycharm高亮。
#todo 原文还有其他兼容性和优化，比如使用Xavier初始化和可选择激活函数，回头再调 #他甚至有L2正则化
import numpy as np
from chap05_affine_layer import Affine
from chap05_relu_layer import ReluLayer
from chap05_softmax_layer import SoftmaxWithLoss
from collections import OrderedDict
class MultiLayerNet():
    def __init__(self, lr = 0.001,input_size = 784, hidden_units_list = [],output_size = 10, weight_init_std = 0.01):#做些前置工作，超参数、参数的初始化等等#每一层的unit数呢？一个list？
        self.lr = lr
        self.params = {}
        self.input_size,self.hidden_units_list,self.output_size = input_size,hidden_units_list,output_size
        self.__init_weight(weight_init_std = weight_init_std)#单独写一个函数初始化weight。
        self.hidden_layer_nums = len(hidden_units_list)#这个其实就是中间层数？all_size_list虽然加了头尾，不过是从1开始的，4+1+1-1=5，这里是4，对吗？不太一致！他可能用了隐层不包含输出层的计数方法

        #成对实现affine+relu#用什么串联？tensor？input？这不是框架，不需要，用一个保存层的dict，然后从里边逐步的取？在predict的时候才拼凑流程
        self.layers = OrderedDict()#用ordereddict？
        for i in range(self.hidden_layer_nums):#用谁迭代？params的数量是affine层的双倍，用i，用key？愚蠢的方法，他重新拼凑了一次W和b。。。。
            self.layers['Affine' + str(i+1)] = Affine(self.params['W'+str(i+1)],self.params['b' + str(i+1)])
            self.layers['ReLU' + str(i+1)] = ReluLayer()
        idx = self.hidden_layer_nums + 1#todo 他这种混乱写法，这个for循环要比all_size_list小两个，然后还得手动+1，真不敢恭维
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()#是softmax层的类，不是softmax函数

        pass
    def __init_weight(self,weight_init_std = 0.01):#之前的两层网络，W1、W2都是手动指定的，这里打算怎么写？
        all_size_list = [self.input_size] + self.hidden_units_list + [self.output_size]#他用这么一个概念，拼凑了一次，至于这个函数，参数不用传了，用，self
        #但是头部进来了，尾部呢？一个for循环，两边都照顾还是麻烦，改range起始位置
        for i in range(1,len(all_size_list)):
            self.params['W' + str(i)] = weight_init_std * np.random.randn(all_size_list[i-1],all_size_list[i])#todo 参考代码使用了Xavier，回头考虑
            self.params['b' + str(i)] = np.zeros(all_size_list[i])
    def predict(self,x):#循环一个前向传播，不接softmax，裸结果
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    def loss(self,x,t):#用到predict
        x = self.predict(x)#这里，返回值叫x叫y无所谓吧，至少对于forward参数命名来说，还叫x
        loss = self.last_layer.forward(x,t)
        return loss

    def gradient(self,dout):#这里不管后边softmax和cross entropy的事，直接接受dout就行了，那么哪个接口调用它着？loss接口？train接口？目前train都在外边。目前流程都是拿到w，拿到grad，手动在外部更新参数。
        layers = self.layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)


        #最后return的是gradient，但是途中有个中转，从各层的grad拿到self的grads，这一步self转存以后什么必要，没有必要，2层网络原码这里只是为了拼一个结构好返回，没存到self
        grads = {}#具体到每一层的每个key，之前是手写的，现在迭代循环了，需要用到了。而且命名也变了，直接是对外直接暴露的W1和b1，现在是每层一个W和b
        # grads['W1'] = self.layers['Affine1'].dW
        grads[]



    def accuracy(self):
        pass

if __name__ == '__main__':
    from book_dir.dataset.mnist import load_mnist
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=True)
    # hidden_layer_nums = 4
    hidden_units_list = [100,100,100,100]
    net = MultiLayerNet(lr = 0.001, input_size = 784,output_size = 10, hidden_units_list = hidden_units_list)
    predict = net.predict(x_train[0])


