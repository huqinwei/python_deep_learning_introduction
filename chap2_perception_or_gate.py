import numpy as np
def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([1.0,1.0])
    b = -0.9
    res = np.sum(w*x) + b
    if res > 0:
        return 1
    else:
        return 0
if __name__ == '__main__':
    print(OR(1, 1))
    print(OR(1,0))
    print(OR(0,0))



