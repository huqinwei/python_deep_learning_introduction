from chap04_grad_2d import numerical_gradient
import matplotlib.pyplot as plt
import numpy as np
def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x_history = []
    x = init_x
    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f,x)
        x -= lr * grad
    return x,np.array(x_history)
def function_2(x):
    return x[0]**2 + x[1] ** 2
init_x = np.array([-3.0,4.0])

# lr = 10.0
# lr = 1e-10
lr = 0.1
x,x_history = gradient_descent(function_2,init_x,lr=lr,step_num=100)

print(x)
print(x_history.shape)
print(x_history[0:4,0:2])
print(x_history.T[0:2,0:4])
X,Y = np.meshgrid(x_history.T[0],x_history.T[1])
print(X.shape,Y.shape)##差一步绘制3D的接口,
# plt.scatter(X,Y,1)#其实又没有value，光有横纵坐标也不是想象的图,况且横纵坐标画格之后，点更多了。。。
# plt.scatter(X,Y,function_2(np.array([X,Y])))
# plt.plot()

#象限:起终点 坐标，分开表示x、y轴的起终点
plt.plot( [-5, 5], [0,0], '--b')
# plt.plot( [-5, 5], [0,2], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')#全部的起点和全部的终点，因为是分开的，所以就是绘制全部的点，用plot

plt.show()
