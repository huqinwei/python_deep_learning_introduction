from chap05_NN_2layer import TwoLayerNet
from book_dir.dataset.mnist import load_mnist
import numpy as np
import matplotlib.pyplot as plt

#hyper params
lr = 0.001
batch_size = 100
iterations = 20000
print_iterations = 100
#其他的比如hidden这些不废话了，直接写死了

#init net
net = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

#load data
((x_train,t_train),(x_test,t_test)) = load_mnist(normalize = True, flatten = True, one_hot_label = True)


loss_history = []
accuracy_history = []

#iterate train
for i in range(iterations):
    mask = np.random.choice(x_train.shape[0],batch_size)
    x_batch = x_train[mask]
    t_batch = t_train[mask]
    grads = net.gradient(x_batch,t_batch)
    for k in net.params:
        net.params[k] -= lr * grads[k]
    if i % print_iterations == 0:
        loss = net.loss(x_batch,t_batch)#暂时用batch来打印一下
        loss_history.append(loss)
        accuracy = net.accuracy(x_batch,t_batch)
        accuracy_history.append(accuracy)


pred = np.argmax(net.predict(x_test[:10]),axis = 1)
label = np.argmax(t_test[:10],axis = 1)
print(pred == label)

iters_plot = np.arange(iterations / print_iterations)
plt.plot(iters_plot,loss_history, label='train acc')
plt.plot(iters_plot,accuracy_history, label='train loss')
plt.xlabel("iterations")
plt.ylabel("loss&accuracy")
plt.legend()
plt.show()


# #
# net.accuracy()
# net.loss()
# net.predict()





