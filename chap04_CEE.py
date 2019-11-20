import numpy as np
#这书有点误导人，这俩函数自动就把t从onehot变成了数字，第一个旧函数是针对onehot target的，第二个新函数是针对纯数字的,虽然他最后是解释了，但是当初没说.
def cross_entropy_error_one_hot_1d(y,t):
    delta = 1e-7
    # print('y:',y)
    # print('t:',t)
    return -np.sum(t * np.log(y+delta))
def cross_entropy_error(y,t):
    if y.ndim == 1:
        y = y.reshape(1,y.size)
        t = t.reshape(1,t.size)
    delta = 1e-7
    batch_size = y.shape[0]#after reshape
    # print('y:',y)
    # print('t:',t)
    # print('np.arange(batch_size):',np.arange(batch_size))
    # print('y[arange]:',y[np.arange(batch_size)])
    # print('y[arange,t]:',y[np.arange(batch_size),t])
    # # print('y[[0],t]:',y[[0],t])
    # print('log y[arange,t]:',np.log(y[np.arange(batch_size),t]))
    # print('same:',np.log(y[np.arange(batch_size)],t))
    # print('same:',t * np.log(y[np.arange(batch_size)]))

    # return -np.sum(t*np.log(y+delta)) / batch_size
    return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size
def cross_entropy_error_one_hot(y,t):
    if y.ndim == 1:
        y = y.reshape(1,y.size)#change to 2-d
        t = t.reshape(1,t.size)
    delta = 1e-7
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7)) / float(batch_size)

if __name__ == '__main__':
    # t = [0,0,1,0,0,0,0,0,0,0]
    # y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
    # # print(np.log(y+ 1e-7))#TypeError: can only concatenate list (not "float") to list
    # # print(np.log(np.array(y) + 1e-7))#this works
    # mse = cross_entropy_error(np.array(y),np.array(t))##parameters t and y swapped
    # print(mse)
    # y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
    # mse = cross_entropy_error(np.array(y),np.array(t))
    # print(mse)
    # parameters orders!in cross entropy?only one '1' in t,but y all have values
    #0.5  6.4        2.3 12

    #vectorization

    dim_1 = False
    if dim_1:
        y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
        t = t=2#[0,0,1,0,0,0,0,0,0,0]
    else:
        y = [[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0], [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]]
        t = [2,2]#[[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
    nd_y = np.array(y)
    print(nd_y.shape,nd_y.ndim,nd_y.size)
    if nd_y.ndim == 1:
        reshape_nd_y = nd_y.reshape(1,len(nd_y))
        print(reshape_nd_y)
        reshape_nd_y = nd_y.reshape(1,nd_y.size)
        print(reshape_nd_y)
        print('origin unchanged,',nd_y)
    # print('old ce:',cross_entropy_error_one_hot(np.array(y),np.array(t)))
    print('ce:',cross_entropy_error(np.array(y),np.array(t)))