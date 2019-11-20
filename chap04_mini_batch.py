import numpy as np

if 0:#choice demo
    np.random.seed(0)#make sure every time runs same
    switch_to_arange = False
    if not switch_to_arange:
        print(np.random.choice(5,3))
    else:
        print(np.random.choice(np.arange(5),3))
from book_dir.dataset.mnist import load_mnist

(x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=True)
print(x_train.shape,t_train.shape)
print(x_train.shape[0],len(x_train))
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size,batch_size)#or batch index?
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(t_batch)
from chap04_CEE import cross_entropy_error,cross_entropy_error_old
cross_entropy_error(,t_batch)



