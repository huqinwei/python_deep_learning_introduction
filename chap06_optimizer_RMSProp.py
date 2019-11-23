#Adagrad的改进版，leaky版

import numpy as np
# from chap06_optimizer_SGD import SGD
class RMSProp():
    def __init__(self,lr = 0.01,decay_rate = 0.9):
        self.lr = lr
        self.h = None
        self.decay_rate = decay_rate

    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            for k,val in grads.items():
                self.h[k] = np.zeros_like(val)
        for k in self.h.keys():
            self.h[k] *= self.decay_rate
            self.h[k] += (1 - self.decay_rate) * grads[k] ** 2#
            params[k] -= self.lr  * grads[k] / (np.sqrt(self.h[k]) + 1e-7)#



if __name__ == '__main__':
    from chap05_NN_2layer import TwoLayerNet
    import numpy as np
    from book_dir.dataset.mnist import load_mnist
    import matplotlib.pyplot as plt
    from chap06_optimizer_SGD import SGD
    from chap06_optimizer_Momentum import Momentum
    from chap06_optimizer_Adagrad import Adagrad


    iterations = 3000
    batch_size = 256
    lr = 0.01

    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=True)


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



    iterations_plot = np.arange(iterations)
    plt.plot(iterations_plot,loss_history_Adagrad,label='Adagrad')
    plt.plot(iterations_plot,loss_history_RMSProp,label='RMSProp')
    plt.plot(iterations_plot,loss_history_Momentum,label='Momentum')
    plt.plot(iterations_plot,loss_history_SGD,label='SGD')
    plt.legend()
    plt.show()







