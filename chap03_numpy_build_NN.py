import numpy as np
import chap3_sigmoid_function

def identity_function(x):#output layer activation
    return x

X = np.array([1.0,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])
print(X.shape)
print(W1.shape)
print(B1.shape)

A1 = np.dot(X,W1) + B1
print(A1.shape)
print(A1)

Z1 = chap3_sigmoid_function.sigmoid_function(A1)
print(Z1)

W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])
print(Z1.shape,W2.shape,B2.shape)
A2 = np.dot(Z1,W2) + B2
print(A2)
Z2 = chap3_sigmoid_function.sigmoid_function(A2)
print(Z2)

W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])
A3 = np.dot(Z2,W3) + B3
Z3 = identity_function(A3)
print(A3,Z3)

