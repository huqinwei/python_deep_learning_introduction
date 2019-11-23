import numpy as np
import matplotlib.pyplot as plt
#测试hist接口,虽然还不够详细，但是能大概解释chap06_weight_init_activation_histogram.py上边那句话了，画四个图，每个图都是一个层的hist，hist是统计，a展开以后统计分布，30是分的细致程度，(0,1)是范围


i = 4
j = 5

a = np.arange(1,10)
b = np.arange(10,20)
c = np.array([10 for i in range(10)])

plt.subplot(i,j,1)
plt.hist(b,10)
plt.subplot(i,j,2)
plt.hist(b,30)
plt.subplot(i,j,3)
plt.hist(b,3)
plt.subplot(i,j,4)
plt.hist(c,bins=4)#第二个参数可以理解为，分几格

plt.subplot(i,j,5)
plt.hist(a,label='origin a')
plt.legend()

plt.subplot(i,j,6)
plt.hist(a,bins=30,label='use bins')
plt.legend()

# 说明是如果bins是sequence，则range无效，但是看起来目前是有效的？什么样的bins是序列的？
# 并没有说bins和range会冲突，是说bins和range会影响自动缩放，变成指定缩放?todo 说了，range会受bin影响，Range has no effect if *bins* is a sequence.
plt.subplot(i,j,7)
plt.hist(a,30,range=(0,1),label = 'use range and bins')
plt.legend()

plt.subplot(i,j,8)
plt.hist(a,range=(0,1),label = 'use range')
plt.legend()

#尝试一个不再范围内的
plt.subplot(i,j,9)
plt.hist(a,range=(0,0.9),label = 'out of range')
plt.legend()

plt.subplot(i,j,10)
plt.hist(a,range=(0,5.0),label = 'fake half in range')#左闭右开
plt.legend()

plt.subplot(i,j,11)
plt.hist(a,range=(0,5.1),label = 'real half in range')#左闭右开
plt.legend()

plt.subplot(i,j,12)
plt.hist(a,range=(0,10.1),label = 'all in range')#左闭右开
plt.legend()

arr = np.random.randn(100)
plt.subplot(i,j,13)
plt.hist(arr,range=(-1,1),label = '100,')#左闭右开
plt.legend()

arr = np.random.randn(100,100)
plt.subplot(i,j,14)
plt.hist(arr,range=(-1,1),label = '100*100')#左闭右开
plt.legend()

arr = np.random.randn(100,100)
plt.subplot(i,j,15)
plt.hist(arr.flatten(),range=(-2,2),label = 'flatten')#左闭右开
plt.legend()
arr = np.random.randn(100,100)
plt.subplot(i,j,16)
plt.hist(arr.flatten(),bins=30,range=(-2,2),label = 'flatten')#左闭右开
plt.legend()


#为何左开右闭？这就是原因，因为柱状图的一个bin是一个区间，那么第一个区间就是[-1,-0.75),然后[-0.75,-0.5)，
# todo 最后呢，(3.75,4)但是不包含4?但是他好像包含了，如果4.0包含在左边，那么-1.0呢？好像确实有歧义了。todo 仔细看说明，右边是特殊的
'''All but the last (righthand-most) bin is half-open.  In other
            words, if `bins` is::

                [1, 2, 3, 4]

            then the first bin is ``[1, 2)`` (including 1, but excluding 2) and
            the second ``[2, 3)``.  The last bin, however, is ``[3, 4]``, which
            *includes* 4.
'''
arr = np.array([-1.0,-0.8,-0.75,-0.70,-0.6,-0.45,0.0,0.1,0.2,1.1,1.2,3.3,3.4,4.0])
plt.subplot(i,j,17)
plt.hist(arr.flatten(),bins=20,range=(-1,4),label = 'test range')#最后一个是全闭的
plt.legend()

#原来bins可以这样用，sequence就是说一个序列，那么bins本来就指定了范围,并且不用等间隔
arr = np.array([-1.0,-0.8,-0.75,-0.70,-0.6,-0.45,0.0,0.1,0.2,1.1,1.2,3.3,3.4,4.0])
plt.subplot(i,j,18)
plt.hist(arr.flatten(),bins=[-2,-1,0,1,2,3.9,4],range=(-1,4),label = 'test range')#最后一个是全闭的
plt.legend()
#跑完上边这两个，至少对于relu的激活问题，可以确定，坐标轴数字0右侧的，其实不是0，只是很小的值！！！
# 也不全对，当你看不到其他区间有分布的时候（表达消失），要记住,这个0开头的区间包含了大于0的值，但是当你找不到它左边的区间有分布的时候，要记得，它是包含0本身的


plt.show()