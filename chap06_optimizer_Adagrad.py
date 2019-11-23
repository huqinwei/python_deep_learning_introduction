#Adaptive gradient
#自适应，指的是每个维度的梯度（学习率）自适应
#实现思路，实现思路上，和Momentum差不多，每个维度肯定都要维护它对应的变量，只要把Momentum的公式替换成Adagrad就行了。
#momentum的v等于梯度的融合，所以要-梯度，这里的h = dW**2其实是累加，用加法
#理论上，这个方法，也不是dW**2是二次导数，为了计算成本，简化版本。

#自己先跑一遍，出意外了，刚才想着分母的事，结果忘了加了，报错了

import numpy as np
# from chap06_optimizer_SGD import SGD
class Adagrad():
    def __init__(self,lr = 0.01):
        self.lr = lr
        self.h = None

    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            for k,val in grads.items():#.items用法
                #两种写法都行
                if 0:
                    self.h[k] = np.zeros_lize(grads[k])
                else:
                    self.h[k] = np.zeros_like(val)
        for k in self.h.keys():
            self.h[k] += grads[k] ** 2#分母不能加学习率
            params[k] -= self.lr  * grads[k] / (np.sqrt(self.h[k]) + 1e-7)#分母防0



if __name__ == '__main__':
    from chap05_NN_2layer import TwoLayerNet
    import numpy as np
    from book_dir.dataset.mnist import load_mnist
    import matplotlib.pyplot as plt
    from chap06_optimizer_SGD import SGD
    from chap06_optimizer_Momentum import Momentum


    iterations = 20000
    batch_size = 256
    lr = 0.01

    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=True)
    net = TwoLayerNet(784,50,10)
    optimizer = Adagrad(lr)
    loss_history = []
    for i in range(iterations):
        mask = np.random.choice(x_train.shape[0],batch_size)
        x_batch,t_batch = x_train[mask],t_train[mask]
        grads = net.gradient(x_batch,t_batch)
        optimizer.update(net.params,grads)
        loss = net.loss(x_batch,t_batch)
        loss_history.append(loss)

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

    # net = TwoLayerNet(784, 50, 10)
    # loss_history_SGD = []
    # optimizer = SGD(lr)
    # for i in range(iterations):
    #     mask = np.random.choice(x_train.shape[0],batch_size)
    #     x_batch,t_batch = x_train[mask],t_train[mask]
    #     grads = net.gradient(x_batch,t_batch)
    #     optimizer.update(net.params,grads)
    #     loss = net.loss(x_batch,t_batch)
    #     loss_history_SGD.append(loss)



    iterations_plot = np.arange(iterations)
    plt.plot(iterations_plot,loss_history,label='Adagrad')
    plt.plot(iterations_plot,loss_history_Momentum,label='Momentum')
    # plt.plot(iterations_plot,loss_history_SGD,label='SGD')
    plt.legend()
    plt.show()







