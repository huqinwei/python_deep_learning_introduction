#这个还是要好好练一下的，手写，各种激活的打印和监视。

# from book_dir.common.multi_layer_net import MultiLayerNet#没用上！！！！！！
import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
def tanh(x):#自己平移sigmoid做的tanh和库里的tanh，和网上标准公式实现的tanh，是否有区别，
    return (2.0 / (1 + np.exp(-2*x))) - 1
def tanh2(x):#公式推导，其实是等价的
    y=(math.e**(x)-math.e**(-x))/(math.e**(x)+math.e**(-x))
    return y
def relu(x):
    return np.maximum(0,x)

x = np.random.randn(1000,100)
node_num = 100
hidden_layer_size = 5
activations = {}
before_activations = {}
weights = {}
hist_demo = False
plain_weight_init = False
activation = 'relu'

for i in range(hidden_layer_size):
    if i != 0:#有一层没算
        x = activations[i-1]#那么第一层呢，直接用原定义，注意这是裸写的网络，不是类！

    w = np.random.randn(node_num, node_num)
    if plain_weight_init:
        weight_init_std = 0.01#1,0.1,0.01,0.001#量级越小，后边就越消失
        w = w * weight_init_std
        #todo(高亮) 为什么relu用weight_init_std=1的激活分布也很小？？？？？？用0.1就分布均匀一些？因为更多在负轴？不应该无论方差大小，正负都是均衡的么？可以打印一下激活前的观察。
        #todo 三种情况，std=1，因为方差大，；；0.01的时候，因为分布就集中于零，所以激活肯定集中于零。0.1的时候，相比其他两种情况，反倒均衡一些？说不过去吧！
        #todo 其实并没有这么大差距？视觉欺骗，仔细看纵坐标，都差不多,但是0.1因为比1更集中，在深层没有那种分布朝向几十几百的分散效应，显得好看
        # 而std=1，因为都分到更大的值，所以坐标轴大，但是大更的值的分布少，显得集中一些罢了，也就不好看，但是只是形状，仔细看，是因为几十几百的区间比较空罢了
        # 如果简单看形状，好像都差不多，0.01也不难看，但是仔细看区间分布，其实0.01的激活值，横坐标非常小
    else:#怎么理解Xavier，粗暴的理解就是，有多少个输出，就用根号几当分母，输出越多越需要缩小？越缩小量级不是越集中么？
        #可能应该回到最初，在没有weight_init_std的时候，分布是向两边分散的，这个初始化问题是一个梯度消失和表达消失的trade-off。
        # 如果weight_init_std=1，这个时候是梯度消失趋势，所以集中是好的，后边看到的那些weight_init_std=0.01的时候，是中间高两边低，那才是表达能力消失。
        # 所以在梯度消失的状态下，缩小应该是有集中趋势。个人感觉这一层的节点也有关系，因为节点越多，分化（复制）出来的数也就越多，论文先不看了，有了BN，感觉这不是一个很需要深究的问题。

        if activation == 'tanh':
            w = w / np.sqrt(node_num)
        elif activation == 'sigmoid':
            w = w / np.sqrt(node_num)
        elif activation == 'relu':#作为对比，relu最好也用weight_init_std=0.01跑一次
            w = (w / np.sqrt(node_num))# * np.sqrt(2)#可以注释掉根号2对比，差距明显，这个很好解释，因为relu只有正半轴

        weights[i] = w

    z = np.dot(x,w)
    if activation == 'tanh':
        a = tanh(z)
    elif activation == 'sigmoid':
        a = sigmoid(z)
    elif activation == 'relu':
        a = relu(z)
    else:
        a = sigmoid(z)
    activations[i] = a
    before_activations[i] = z

layer_nums = len(activations)
for i,w in weights.items():
    plt.subplot(3,layer_nums,i+1)#i从0起，plot从1起
    plt.title(str(i+1) + "-layer")
    wf = w.flatten()

    if activation == 'tanh':
        plt.hist(wf,label='weights')#用tanh可以看到分布更好看一些，钟形
    elif activation == 'sigmoid':
        plt.hist(wf,label='weights')
    elif activation == 'relu':
        #限定range在0到1，对比会更明显,如果不限定，虽然形状相似，横轴坐标也有不同。
        plt.hist(wf,label='weights')#根据hist的功能，0右侧是大于0，那么relu不可能0左侧没东西吧？0不要面子的吗？？？？记错了，左闭右开，那么其实是[0,0.x]，其实是包含0的,也包含极小的非零值
plt.legend()

for i,z in before_activations.items():
    plt.subplot(3,layer_nums,i+1 + layer_nums)#i从0起，plot从1起
    plt.title(str(i+1) + "-layer")
    zf = z.flatten()

    if activation == 'tanh':
        plt.hist(zf,30,label='z')#用tanh可以看到分布更好看一些，钟形
    elif activation == 'sigmoid':
        plt.hist(zf,30,label='z')
    elif activation == 'relu':
        plt.hist(zf,label='z')
plt.legend()

for i,a in activations.items():
    plt.subplot(3,layer_nums,i+1 + layer_nums + layer_nums)#i从0起，plot从1起
    plt.title(str(i+1) + "-layer")
    af = a.flatten()

    if activation == 'tanh':
        plt.hist(af,30,range=(-1,1),label='activations')#用tanh可以看到分布更好看一些，钟形
    elif activation == 'sigmoid':
        plt.hist(af,30,range=(0,1),label='activations')
    elif activation == 'relu':
        #限定range在0到1，对比会更明显,如果不限定，虽然形状相似，横轴坐标也有不同。
        plt.hist(af,range=(0,1),label='activations')#根据hist的功能，0右侧是大于0，那么relu不可能0左侧没东西吧？0不要面子的吗？？？？记错了，左闭右开，那么其实是[0,0.x]，其实是包含0的,也包含极小的非零值
plt.legend()



plt.show()


# #自己平移sigmoid做的tanh和库里的tanh，和网上标准公式实现的tanh，是否有区别？等价的！
# x = np.arange(-5,5,0.1)
# y = tanh(x)
# plt.subplot(1,4,1)
# plt.plot(x,y)
#
# y2 = np.tanh(x)
# plt.subplot(1,4,2)
# plt.plot(x,y2)
#
# y3 = tanh2(x)
# plt.subplot(1,4,3)
# plt.plot(x,y3)
#
#
# y4 = [math.tanh(i) for i in x]
# plt.subplot(1,4,4)
# plt.plot(x,y4)
# plt.show()







