import numpy as np

def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)#no need to multiply 0.5,and it comsumes computation


if __name__ == '__main__':

    x = np.array([1.0,2.0])
    print(np.square(x))
    print(x**2)
    print(mean_squared_error(x,0))
    x2 = np.array([0.0,0.0])
    print(mean_squared_error(x,[0.0,0.0]))
    print(mean_squared_error(x,x2))


    t = [0,0,1,0,0,0,0,0,0,0]
    y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
    mse = mean_squared_error(np.array(y),np.array(t))#parameters orders wrong!this is bad in cross entropy?yes,because only one '1' in t
    print(mse)
    y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
    mse = mean_squared_error(np.array(y),np.array(t))
    print(mse)
    # mse = mean_squared_error(t,y)#unsupported operand type(s) for -:'list' and 'list'
    # print(mse)

