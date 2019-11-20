from chap04_NN_2layer import TwoLayerNet
import numpy as np
from book_dir.dataset.mnist import load_mnist
import matplotlib.pyplot as plt

(x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=True)
train_loss_list = []
train_acc_list = []
test_acc_list = []

# print(x_train.shape)
# print(t_train.shape)

nn = TwoLayerNet(784,50,10)


lr = 0.1
iter_num = 10#000
train_size = x_train.shape[0]
batch_size = 10#0

# batch_size = 99
iter_per_epoch = max(int(train_size / batch_size),1)#不严谨了，原代码没有int()

for i in range(iter_num):
    mask = np.random.choice(train_size,batch_size)

    x_batch = x_train[mask]
    t_batch = t_train[mask]

    grads = nn.numerical_gradients(x_batch,t_batch)

    for key in nn.params:
        nn.params[key] -= grads[key] * lr
    loss = nn.loss(x_batch,t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = nn.accuracy(x_train,t_train)##这也不严谨，他train loss实际是batch，这里acc是整个train set,at least,not consistent
        test_acc = nn.accuracy(x_test,t_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc,test_acc)

    # nn.params -= grads
plt.plot(np.arange(iter_num),train_loss_list)
plt.show()



