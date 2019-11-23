#不照搬代码，先分析：
#要领：维护一个v，把梯度更新到v上，v带衰减系数，初始化给进去学习率和衰减
#和SGD显著不同，这里grads是多个key分别操作的，所以v呢？多个v？
# 如果是多个，在初始化的时候怎么确定？而且你总得变得吧？而且这个优化器是一次次调用的，而且还需要保留状态。如果每次给的params和grads不一样呢？
#不是一定不能实现，主要看他想怎么实现，比如“一次定型”，说这个优化器就针对这些params，那么init就应该传入这些params，反而update都不用传params了。
#看了书上代码，他确实是假定一次成型，但是这一次是在update中，第一次update传入什么，就固定住
#总之，书上的版本也不是一个工程版本，算是一个简化的示意版本吧，就是在约定好的条件下能用，但是绝对不通用。
import numpy as np
class Momentum():
    def __init__(self,lr = 0.01,momentum = 0.9):
        self.lr = lr
        self.v = None#先不考虑初始优化问题，vanilla版本
        self.momentum = momentum

    def update(self,params,grads):
        if self.v is None:
            self.v = {}
            for k,val in grads.items():#.items用法
                # self.v[k] = grads[k]#写错了，初始化不是把grad赋值过去，只是做一个shape相同的0集
                #两种写法都行
                if 0:
                    self.v[k] = np.zeros_lize(grads[k])
                else:
                    self.v[k] = np.zeros_like(val)
        for k in self.v.keys():
            # self.v[k] -= self.lr * grads[k]#写错了，这还是SGD，而且-=的写法一步写不出来动量衰减，必须用正常=
            self.v[k] = self.v[k] * self.momentum - self.lr * grads[k]
            params[k] += self.v[k]#一致性，只要v对grad都是减法，这里就是加法






if __name__ == '__main__':
    from chap05_NN_2layer import TwoLayerNet
    import numpy as np
    from book_dir.dataset.mnist import load_mnist
    import matplotlib.pyplot as plt
    from chap06_optimizer_SGD import SGD


    iterations = 1000
    batch_size = 256
    lr = 0.01

    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=True)
    net = TwoLayerNet(784,50,10)
    optimizer = Momentum(lr)
    loss_history = []
    for i in range(iterations):
        mask = np.random.choice(x_train.shape[0],batch_size)
        x_batch,t_batch = x_train[mask],t_train[mask]
        grads = net.gradient(x_batch,t_batch)
        optimizer.update(net.params,grads)
        loss = net.loss(x_batch,t_batch)
        loss_history.append(loss)

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
    plt.plot(iterations_plot,loss_history,label='Momentum')
    plt.plot(iterations_plot,loss_history_SGD,label='SGD')
    plt.show()




