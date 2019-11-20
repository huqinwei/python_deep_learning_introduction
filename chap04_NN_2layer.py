import numpy as np
from chap03_sigmoid_function import sigmoid_function as sigmoid
from chap03_softmax_function import softmax
from chap04_numerical_diff import numerical_diff
from chap04_numerical_gradient import numerical_gradient
from chap04_CEE import cross_entropy_error,cross_entropy_error_one_hot

class TwoLayerNet():
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size,hidden_size) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size,output_size) * weight_init_std
        self.params['b2'] = np.zeros(output_size)

    def predict(self,x):
        a1 = np.dot(x,self.params['W1']) + self.params['b1']
        z1 = sigmoid(a1)
        a2 = np.dot(z1,self.params['W2']) + self.params['b2']
        y = softmax(a2)
        return y

    def loss(self,x,t):
        y = self.predict(x)
        loss = cross_entropy_error_one_hot(y,t)
        return loss

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)
        acc = np.sum(y==t) / float(x.shape[0])
        return acc

    #import grad

    def numerical_gradients(self,x,t):
        #这里不能用self.loss，谨记，传进去的f，是一个给出w就能得到loss的函数,而self.loss，实际上是个float的数据
        self.grads = {}
        loss_W = lambda W:self.loss(x,t)#dLoss/dW
        self.grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        self.grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        self.grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        self.grads['b2'] = numerical_gradient(loss_W,self.params['b2'])

        return self.grads




if __name__ == '__main__':


    # x = np.array([0.3,0.4])
    # t = np.array([0,0,1])
    # net = TwoLayerNet(2,3,3)
    # print(net.gradients(x,t))

    mnist_net = TwoLayerNet(784,100,10)
    for param in mnist_net.params:#for i,key in enumerate(mnist_net.params):
        print(mnist_net.params[param].shape)

    x = np.random.randn(100,784)
    t = np.random.randn(100,10)
    y = mnist_net.predict(x)
    print('y :',np.argmax(y,axis=1))

    grads = mnist_net.numerical_gradients(x,t)



    try:#没求过gradients就没有
        if not mnist_net.grads:#try?
            print('grads not initialized!!!!')
        else:
            for g in mnist_net.grads:
                print(mnist_net.grads[g].shape)
    except AttributeError as e:
        print('except:',e)
    finally:
        print('final')






