import numpy as np

def AND(x1,x2):
    w1 = 0.5
    w2 = 0.5
    theta = 0.7
    res = x1*w1 + x2*w2
    if res > theta:
        return 1
    else:
        return 0
def AND(x1,x2):#replace theta with -b
    w1,w2 = 0.5,0.5
    b = -0.7
    res = x1 * w1 + x2 * w2 + b
    if res > 0:
        return 1
    else:
        return 0


if __name__ == '__main__':

    print(AND(0,0),AND(0,1),AND(1,0),AND(1,1))

    x = np.array([0,1])
    x = np.array([1,1])
    w = np.array([0.5,0.5])
    print(x)
    print(w)
    print(w*x)
    sum = np.sum(w*x)
    print(sum)
    b = -.7
    print(sum + b)


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([.5,.5])
    b = -.7
    res = np.sum(w*x) + b
    if res > 0:
        return 1
    else:
        return 0

if __name__ == '__main__':
    print(AND(0,1))
    print(AND(1,1))








