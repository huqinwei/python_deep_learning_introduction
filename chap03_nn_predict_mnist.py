import numpy as np
from book_dir.dataset.mnist import load_mnist
import pickle
from chap3_sigmoid_function import sigmoid_function as sigmoid
from chap3_softmax_function import softmax
import time
def get_data():
    # (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=False)
    _,(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test,t_test

def init_network():
    # with open("book_dir//dataset//mnist.pkl",'rb') as f:#wrong file,this is data
    with open("book_dir//ch03//sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
    return network
def predict(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    A1 = np.dot(x,W1) + b1
    Z1 = sigmoid(A1)
    A2 = np.dot(Z1,W2) + b2
    Z2 = sigmoid(A2)
    A3 = np.dot(Z2,W3) + b3
    y = softmax(A3)
    print(x.shape,W1.shape,W2.shape,W3.shape)
    print(b1.shape,b2.shape,b3.shape)
    print("A shape:",A1.shape,A2.shape,A3.shape)
    return y


x_test,t_test = get_data()
print(x_test.shape)
network = init_network()
# print(type(network))
# print(network)
print((predict(network, x_test[0])))
print(np.argmax(predict(network, x_test[0])))
print('label:',t_test[0])


if 0:
    sum = 0.0
    for i in range(len(x_test)):
        y = np.argmax(predict(network,x_test[i]))
        if y == t_test[i]:
            sum += 1

    print('accuracy is ',sum / len(x_test))
sum = 0.0
batch_size = 100
# for i in range(0,len(x_test),batch_size):
#     batch_x = x_test[i:i+batch_size]
#如果用了step=batch_size，又用了i*batch*size，100×100的话，第二次循环就会得到空集，按C++就叫越界了
for i in range(0,len(x_test)//batch_size):#keep consistent
    batch_x,batch_t = x_test[i * batch_size:(i+1)*batch_size], t_test[i*batch_size:(i+1)*batch_size]
    pred = predict(network, batch_x)
    y = np.argmax(pred,axis=1)
    res = y == batch_t

    sum += np.sum(res)
print('accuracy is ',sum / len(x_test))

# #这是非并行的，所以中间应该没有什么参数复制N份的操作
# start = time.time()
# res =predict(network, x_test[:10000])
# end = time.time()
# print(res)
# print('time spend ',end-start)
# #1000   time spend  0.002000093460083008
# #10000  time spend  0.023000001907348633
# #5000   time spend  0.009000062942504883







