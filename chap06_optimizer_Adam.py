#应该是有三个变量的优化器，两个惯量系数，简单理解为Momentum和Adagrad（leaky）的结合版，就是RMSProp的分子替换成Momentum
#并且，Momentum也是改版（自身衰减之外，新加入的也有一个系数，类似RMSProp），不是vanilla版本
#直觉上，不太好解释？因为Adagrad直觉上的解释是，一次导数和二次导数累加的对比，这里分子成了惯性，不好解释，直接解释成效果好就行了。
#实现细节上，多维护了一套东西，多加上就好吧

import numpy as np
# from chap06_optimizer_SGD import SGD
class Adam():
    def __init__(self,lr = 0.001,beta1 = 0.9,beta2 = 0.999):
        self.lr = lr
        self.h = None
        self.v = None
        self.iter = 0
        self.beta1 = beta1
        self.beta2 = beta2

    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            for k,val in grads.items():
                self.h[k] = np.zeros_like(val)
        if self.v is None:
            self.v = {}
            for k,val in grads.items():
                self.v[k] = np.zeros_like(val)
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)#Adam会卡在高位，所以只能采用一种学习率衰减？这其实是增加的，log级增加，上限1

        # lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta2 ** self.iter)#
        # lr_t = self.lr * np.sqrt(1.0 - self.beta1 ** self.iter) / (1.0 - self.beta1 ** self.iter)#
        # lr_t = self.lr * np.sqrt(1.0 - self.beta1 ** self.iter) / (1.0 - self.beta2 ** self.iter)

        lr_history.append(lr_t)
        for k in self.h.keys():
            self.v[k] = (self.v[k] * self.beta1) + ((1-self.beta1)*grads[k])#todo 正负号一致性检查
            self.h[k] = (self.h[k] * self.beta2) + ((1 - self.beta2) * grads[k] ** 2)#
            params[k] -= (lr_t  * self.v[k]) / (np.sqrt(self.h[k]) + 1e-7)#

        v_history.append(np.linalg.norm(self.v['W1']))
        h_history.append(np.linalg.norm(self.h['W1']))



if __name__ == '__main__':
    from chap05_NN_2layer import TwoLayerNet
    import numpy as np
    from book_dir.dataset.mnist import load_mnist
    import matplotlib.pyplot as plt
    from chap06_optimizer_SGD import SGD
    from chap06_optimizer_Momentum import Momentum
    from chap06_optimizer_Adagrad import Adagrad
    from chap06_optimizer_RMSProp import RMSProp


    iterations = 300
    iterations_plot = np.arange(iterations)
    batch_size = 256
    lr = 0.001

    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=True)


    net = TwoLayerNet(784,50,10)
    optimizer = Adam(lr)
    loss_history_Adam = []
    lr_history = []
    v_history = []
    h_history = []
    for i in range(iterations):
        mask = np.random.choice(x_train.shape[0],batch_size)
        x_batch,t_batch = x_train[mask],t_train[mask]
        grads = net.gradient(x_batch,t_batch)
        optimizer.update(net.params,grads)
        loss = net.loss(x_batch,t_batch)
        loss_history_Adam.append(loss)

    if 1:

        net = TwoLayerNet(784,50,10)
        optimizer = RMSProp(lr)
        loss_history_RMSProp = []
        for i in range(iterations):
            mask = np.random.choice(x_train.shape[0],batch_size)
            x_batch,t_batch = x_train[mask],t_train[mask]
            grads = net.gradient(x_batch,t_batch)
            optimizer.update(net.params,grads)
            loss = net.loss(x_batch,t_batch)
            loss_history_RMSProp.append(loss)


        net = TwoLayerNet(784,50,10)
        optimizer = Adagrad(lr)
        loss_history_Adagrad = []
        for i in range(iterations):
            mask = np.random.choice(x_train.shape[0],batch_size)
            x_batch,t_batch = x_train[mask],t_train[mask]
            grads = net.gradient(x_batch,t_batch)
            optimizer.update(net.params,grads)
            loss = net.loss(x_batch,t_batch)
            loss_history_Adagrad.append(loss)

        net = TwoLayerNet(784, 50, 10)
        loss_history_Momentum = []
        optimizer = Momentum(lr)
        for i in range(iterations):
            mask = np.random.choice(x_train.shape[0],batch_size)
            x_batch,t_batch = x_train[mask],t_train[mask]
            grads = net.gradient(x_batch,t_batch)
            optimizer.update(net.params,grads)
            loss = net.loss(x_batch,t_batch)
            loss_history_Momentum.append(loss)

        net = TwoLayerNet(784, 50, 10)
        loss_history_SGD = []
        optimizer = SGD(lr)
        for i in range(iterations):
            mask = np.random.choice(x_train.shape[0],batch_size)
            x_batch,t_batch = x_train[mask],t_train[mask]
            grads = net.gradient(x_batch,t_batch)
            optimizer.update(net.params,grads)
            loss = net.loss(x_batch,t_batch)
            loss_history_SGD.append(loss)
        plt.plot(iterations_plot,loss_history_Adagrad,label='Adagrad')
        plt.plot(iterations_plot,loss_history_RMSProp,label='RMSProp',marker='x')
        plt.plot(iterations_plot,loss_history_Momentum,label='Momentum')
        plt.plot(iterations_plot,loss_history_SGD,label='SGD')



    plt.plot(iterations_plot,loss_history_Adam,label='Adam',marker='o')
    if 0:
        plt.subplot(3,1,1)
        plt.plot(iterations_plot,lr_history,label='lr')
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(iterations_plot,v_history,label='v')
        plt.legend()
        plt.subplot(3,1,3)
        plt.plot(iterations_plot,h_history,label='h')
    plt.legend()
    plt.show()







