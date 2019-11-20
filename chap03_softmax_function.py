import numpy as np

def softmax_old(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
def softmax_no_batch(a):#accords to book
    max_a = np.max(a)
    exp_a = np.exp(a - max_a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
def softmax(a):#书上没有给出向量化的代码，这是自己简单写的，原来代码是错的，不针对batch数据
    if a.ndim == 1:
        return softmax_no_batch(a)
    y = np.zeros_like(a,dtype=np.float64)
    for i in range(a.shape[0]):
        y_i = softmax_no_batch(a[i])
        y[i] += np.array(y_i)
    return y

def softmax_batch(a):#自己做一版直接的向量化,效率会geng好么？numpy,whatever
    if a.ndim == 1:
        return 0
    max_a = np.max(a,axis=1)
    max_a = max_a.reshape(max_a.size,1)#根据batch分别做(这个shape是(1,1))
    # exp_a_input = a - max_a#for debug
    exp_a = np.exp(a - max_a)#(1,10)和(1,1)
    sum_exp_a = np.sum(exp_a,axis=1)
    sum_exp_a = sum_exp_a.reshape(sum_exp_a.size,1)#应该还有一个直接增维的
    y = exp_a / sum_exp_a
    return y


if __name__ == '__main__':
    a = np.array([0.3,2.9,4.0])

    exp_a = np.exp(a)
    print(exp_a)
    exp_total = np.sum(exp_a)
    print(exp_total)#doesn't matter

    softmax_a = [i / exp_total for i in exp_a]
    print(softmax_a)
    print(np.sum(softmax_a))

    y = exp_a / exp_total#paralize
    print(y)

    # a = np.array([1010,1000,990])
    # print(softmax_old(a))
    # print(softmax(a))




    #对比，自己的np版本比书上的for循环版本会快一些，
    #a的形状是100×1000时，4倍时间
    #1000*1000，2.6times
    #10000*100,loop10,1.44times
    #10000*100,loop50,11times
    #可能sum和max之类的都要时间，不好估算，反正比那个快就是了
    # 外部的for循环不可避免，所以k不同，倍数变化很大，应该主要变batch_size
    # k越小，i越大，j越小，效率差距越明显
    # 但是k在最外层，按理说不影响，是同步改变倍数吧？主要是相应的改变了i和j,机器性能瓶颈，随意调的
    #大概100倍封顶？可能和我的机器有关
    import time
    k = 3
    i = 640000
    j = 5
    a = np.random.randn(i,j)
    start = time.time()
    for i in range(k):
        softmax(a)
    end = time.time()
    print('spend:',end-start)

    start = time.time()
    for i in range(k):
        softmax_batch(a)
    end = time.time()
    print('spend:',end-start)







