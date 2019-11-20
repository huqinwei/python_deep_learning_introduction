#对比ndarray和list的数据类型变化，结论是ndarray固定，至少是不能通过给元素赋值的途径更改
#不能强制转换，要用astype

import numpy as np
x0 = 3
x1 = 4
x = [x0,x1]
nd_x = np.array(x)

x2 = 3.0
x3 = 4.0
xx = [x2,x3]
nd_xx = np.array(xx)

print(type(x),type(x[0]))
print(type(xx),type(xx[0]))
print(type(nd_x),nd_x.dtype)
print(type(nd_xx),nd_xx.dtype)

print('change element:')
print('before:',type(x[0]),type(x[1]))
x[0] = x[0] + 0.0001
print('after:',type(x[0]),type(x[1]))#list type changed dynamically,and is element-wise
print('x is ',x)

print('before:',type(xx[0]),type(xx[1]))
xx[0] = xx[0] + 0.0001
print('after:',type(xx[0]),type(xx[1]))#list type changed dynamically,and is element-wise
print('xx is ',xx)

#cant change dtype dynamically,in the other hand,if changed ,how ndarray element to keep consistent with whole ndarray.dtype
print('before:',type(nd_x[0]),type(nd_x[1]),nd_x.dtype,nd_x[0].dtype,nd_x[1].dtype)
nd_x[0] = nd_x[0] + 0.0001
print('after:',type(nd_x[0]),type(nd_x[1]),nd_x.dtype,nd_x[0].dtype,nd_x[1].dtype)
print('nd_x is ',nd_x)

print('before:',type(nd_xx[0]),type(nd_xx[1]),nd_xx.dtype,nd_xx[0].dtype,nd_xx[1].dtype)
nd_xx[0] = nd_xx[0] + 0.0001
print('after:',type(nd_xx[0]),type(nd_xx[1]),nd_xx.dtype,nd_xx[0].dtype,nd_xx[1].dtype)
print('nd_xx is ',nd_xx)
# nd_xx.dtype = np.int32##如果直接改变类型，尤其是从float64改到int32，数据就坏了,这种破坏是不可逆的，也好理解，毕竟我想去操作astype，也按已经更改的dtype去看了，这dtype类似C语言常说的，不应该暴露的细节!todo: why??
# nd_xx.dtype = np.int64#int64 bad too!!!!!!!!!!!!
nd_xx = nd_xx.astype(np.int32)
print('nd_xx is ',nd_xx)
#now nd_xx.dtype is np.int32 and it's fixed.
print('before:',type(nd_xx[0]),type(nd_xx[1]),nd_xx.dtype,nd_xx[0].dtype,nd_xx[1].dtype)
nd_xx[0] = nd_xx[0] + 0.0001
print('after:',type(nd_xx[0]),type(nd_xx[1]),nd_xx.dtype,nd_xx[0].dtype,nd_xx[1].dtype)
print('nd_xx is ',nd_xx)

# grad = numerical_gradient(func_3d,np.array(x))#cant change ndarray.dtype dynamically!!!!!!!!!!!!!!!!!!list can