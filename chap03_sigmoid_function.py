import numpy as np
import matplotlib.pyplot as plt
import chap03_step_function

def sigmoid_function(x):
    y = 1 / (1 + np.exp(-x))
    return y


if __name__ == '__main__':
    x = np.arange(-5.0,5.0,0.1)
    x = np.arange(-8.0,8.0,0.1)
    y = sigmoid_function(x)
    y2 = chap3_step_function.step_function(x)
    plt.plot(x,y)
    plt.plot(x,y2,linestyle='--')#supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    plt.show()














