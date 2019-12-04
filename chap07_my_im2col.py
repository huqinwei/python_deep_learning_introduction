
import numpy as np
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def im2col2(input_data, filter_h, filter_w, stride=1, pad=0):#我自己的实现，为了一种更直觉的表达

    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    # col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    col = np.zeros((N, C, out_h, out_w, filter_h, filter_w))

    for y in range(out_h):
        for x in range(out_w):
            y_start = y*stride
            x_start = x*stride
            col[:,:,y,x,:,:] = img[:,:,y_start:y_start + filter_h, x_start:x_start+filter_w]


    col = col.transpose(0, 2, 3, 1, 4, 5).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):

    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))#这是pad后的形状，因为正向传播时可能就是一个pad的形状，要对应。
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            #从conv变fc，fc卷积，fc反向传播，得到FC形状对应的梯度，梯度从fc形状恢复到conv形状，正向传播就有复制，有重叠，反向对应，所以用+=。
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]#还有pad的位移。