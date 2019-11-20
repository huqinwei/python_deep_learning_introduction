import numpy as np
import chap3_sigmoid_function
def identity_function(x):
    return x
#3->2->2

def init_network():#define parameters
    network = {}#dict
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])#2*3
    network['B1'] = np.array([0.1,0.2,0.3])#1*3
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])#2*2
    network['B2'] = np.array([0.1,0.2])#1*2
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])#2*2
    network['B3'] = np.array([0.1,0.2])#1*2
    return network

def forward(network,x):#define network's layers
    A1 = np.dot(x,network['W1']) + network['B1']
    Z1 = chap3_sigmoid_function.sigmoid_function(A1)
    A2 = np.dot(Z1,network['W2']) + network['B2']
    Z2 = chap3_sigmoid_function.sigmoid_function(A2)
    A3 = np.dot(Z2,network['W3']) + network['B3']
    y = identity_function(A3)

    return y

x = np.array([1,0.5])
print(x[0].dtype)
net = init_network()
print(net)
y = forward(net,x)
print(y)






