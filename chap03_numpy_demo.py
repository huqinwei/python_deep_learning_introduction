import numpy as np

A = np.array([1,2,3,4])
print(A.ndim)
print(A.shape,A.shape[0],type(A.shape))

print(A)

B = np.array([[1,2],[3,4],[5,6]])
print(B)
print(B.ndim)
print(B.shape)

A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
print(A.dot(B))

A = np.array([[1,2,3],[4,5,6]])
B = np.array([[1,2],[3,4],[5,6]])
print(A.ndim,B.ndim,A.shape,B.shape)
print(np.dot(A,B))
print(np.dot(B,A))


A = np.array([[1,2],[3,4],[5,6]])
print(A.shape)
# B = np.array([7,8,9])#matrix to vector,the rule still works
B = np.array([7,8])
print(B.shape,B.ndim)
print(A.dot(B))



X = np.array([1,2])
print(X.shape)
W = np.array([[1,3,5],[2,4,6]])
print(W)
print(W.shape)
print(X.dot(W))











