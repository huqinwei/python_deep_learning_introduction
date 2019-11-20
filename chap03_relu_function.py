import numpy as np
import matplotlib.pyplot as plt
import chap3_step_function
import chap3_sigmoid_function

def relu_function(x):
    # y = np.max(0,x)#maximum of parameter a!!!!!
    y = np.maximum(0,x)
    return y

x = np.arange(-5.0,5.0,0.1)
x = np.arange(-8.0,8.0,0.1)
y = chap3_sigmoid_function.sigmoid_function(x)
y2 = chap3_step_function.step_function(x)
y3 = relu_function(x)
plt.plot(x,y)
plt.plot(x,y2,linestyle='--')#supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
plt.plot(x,y3,linestyle='-.')#supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
plt.show()














