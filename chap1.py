a = [1,2,3,4,5]
b = [i for i in range(1,6)]
print(len(a))
print(a[0])
a[4] = 100
print(a)
print(b)
print(b[:-1])

me = {'height':180}
print(me)
print(me['height'])
me['weight'] = 70
print(me)

hungry = True
sleepy = False
print(type(hungry))
print(not hungry)
print(hungry and sleepy)
print(hungry or sleepy)

if hungry:
    print('I\'m hungry')
    # print(r'i'm hungry')#尽管r能自动转义，'还是需要用"双引"外扩
    print(r"i'm hungry")
    print("i'm hungry")
    print(r"C:\dir1")
    print(r'C:\dir1')

def hello(obj):
    if type(obj) == str:
        print('hello ' + obj + '!')
    else:
        print('hello ' + str(obj) + '!')
hello('cat')
hello(1)

class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized!")
    def hello(self):
        print("Hello " + self.name + "!")
    def goodbye(self):
        print("Good-bye " + self.name + "!")
hu = Man('hu')
hu.hello()
hu.goodbye()

import numpy as np
x = np.array([1.0,2.0,3.0])
y = np.array([2.0,4.0,6.0])
zz = np.zeros((1,3))
zzz = np.zeros((1,4))
print(x)
print(type(x))
print(x.dtype)
print(x[0].dtype)
print(type(x[0]))
print(type(1.0))
print(type(x[0]) == type(1.0))
print(x-y)
z = x + y
print(type(z),z)
print(x * y)
print(x / y)
print(zz)
# print(x / zz)#no error,but inf,RuntimeWarning!!!!

# print(x + zzz)#shape not match

print(x[0].dtype)
# x[0] = 'h'#ValueError: could not convert string to float: 'h'
print(x)

x = np.array([1.0,2.0,3.0])
print(x / 2.0)
print(x.shape)
print(x.dtype)

A = np.array([[1,2],[3,4]])
B = np.array([[10,20]])
print(A*B)
# print(A.dot(B))#valueerror
print(B.dot(A))


#accsess and assignment to matrix
x = np.array([[51,55],[14,19],[0,4]])
print(x)
print(x[0])
y = x[0:1,0:1]#this is a slice
y1 = x[0,0]
y2 = x[0][0]
print(y,y1,y2,type(y),type(y1),type(y2),y.shape,y1.shape,y2.shape)

x[0:1,0:1] = 200
print(x)
x[0:1,0:1] = [[22]]
print(x)
x[0:1,0:2] = [[333,444]]
print(x)
x[0:1,0:2] = 777#broadcast
print(x)
x[0:1,0:2] = 876,765
print(x)
x = x.flatten()
print(x)
print(x[np.array([0,2,4])])
print(x[[0,2,4]])
# print('sth:',x[0,2,4])
print(x[x>15])

import matplotlib.pyplot as plt
x = np.arange(0,6,0.1)
x = np.arange(0,2*np.pi,0.1)
y = np.sin(x)
y2 = np.cos(x)

plt.plot(x,y)
plt.plot(x,y2,linestyle='--',label='cos')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('sin&cos')
plt.legend()#must have label
plt.show()








