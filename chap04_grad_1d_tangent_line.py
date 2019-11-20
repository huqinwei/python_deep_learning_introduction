from chap04_numerical_diff import func3,numerical_diff
import numpy as np

import matplotlib.pyplot as plt
x = np.arange(0.0,20.0,0.1)
y = func3(x)
y1 = 0.2 * x - 0.25#compute this in paper manually:5.0
y2 = 0.3 * x - 1#10.0

def tangent_line(f, x):#y = kx+b
    k = numerical_diff(f, x)
    print('k is ',k)
    b = f(x) - k*x
    return lambda t: k*t + b#just find out k and b,and retuan a function with parameters k,b

tf = tangent_line(func3,5.0)
y3 = tf(x)#compute y wrt. all x

plt.plot(x,y)
plt.plot(x,y1,linestyle='--',color='blue',linewidth='2',label='manual5.0')
plt.plot(x,y2,linestyle='dashed',label='manual10.0')
plt.plot(x,y3,linestyle='-.',color='red',label='tangent_line5.0')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

print(numerical_diff(func3,5,))
print(numerical_diff(func3,10,))
