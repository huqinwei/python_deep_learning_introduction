import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    y = x > 0
    return y.astype(np.int)


if __name__ == '__main__':

    x = np.array([-1.0,0.0,1.0])
    y = x > 0
    print(y)
    print(y.astype(int))
    print(y.astype(np.int))
    print(step_function(x))

    x = np.arange(-5.0,5.0,0.1)
    y = step_function(x)
    plt.plot(x,y)
    plt.ylim(-0.1,1.1)
    plt.show()
    # print(x)
