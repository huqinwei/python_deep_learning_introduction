class SGD():
    def __init__(self,lr = 0.01):
        self.lr = lr

    def update(self,params,grads):
        for key in params.keys():
            params[key] -= grads[key] * self.lr



if __name__ == '__main__':
    from chap05_NN_2layer import TwoLayerNet
    import numpy as np
    from book_dir.dataset.mnist import load_mnist
    import matplotlib.pyplot as plt
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=True)
    net = TwoLayerNet(784,50,10)
    optimizer = SGD(0.01)
    loss_history = []

    for i in range(100):
        mask = np.random.choice(x_train.shape[0],100)
        x_batch,t_batch = x_train[mask],t_train[mask]
        grads = net.gradient(x_batch,t_batch)
        optimizer.update(net.params,grads)
        loss = net.loss(x_batch,t_batch)
        loss_history.append(loss)

    iterations = np.arange(100)
    plt.plot(iterations,loss_history)
    plt.show()




