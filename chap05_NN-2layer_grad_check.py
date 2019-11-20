# gradient check  用到上边的那两个梯度计算，对比得到结果，验证analytic是否正确

import numpy as np
# np.random.seed(1)
import sys,os
sys.path.append('book_dir')  # 为了导入目录的文件而进行的设定
from book_dir.ch05.two_layer_net import TwoLayerNet

from chap05_NN_2layer import TwoLayerNet

from book_dir.dataset.mnist import load_mnist

((x_train,t_train),(x_test,t_test)) = load_mnist(normalize=True,one_hot_label=True)
network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)
x_batch = x_train[:3]
t_batch = t_train[:3]

analytic_gradient = network.gradient(x_batch,t_batch)
numerical_gradient = network.numerical_gradient(x_batch,t_batch)

#todo dict之间直接减是不行的，而且也可能发生歧义，万一你定义的key或者顺序不同呢？(尤其是axis都有了对shape也有要求吧，这个dict有点不好处理逻辑关系)所以还是提取
# avg_diff = np.average(np.abs(analytic_gradient - numerical_gradient),axis = 0)
avg_diff = {}
for k in analytic_gradient:
    print(k)
    # avg_diff[k] = np.average(np.abs(analytic_gradient[k] - numerical_gradient[k]),axis = 0)#todo 不需要axis了，不需要每个dim单独一个值，看整体平均？（保留意见，也许单个方向有错的？具体需求具体分析）
    avg_diff[k] = np.average(np.abs(analytic_gradient[k] - numerical_gradient[k]))
# print(avg_diff)
print(avg_diff)

#用自己的模型，有过每个都1e-6,1e-7，有过0.1~0.01


#很难验证？seed？因为weight是随机的，尽管x_batch一样，每次结果都不同,np seed锁定
#自己的{'W1': 0.010388227309640195, 'b1': 0.07345963770673933, 'W2': 0.013504817355636295, 'b2': 0.029683413130214387}
#他的{'W1': 4.4915425166251315e-10, 'b1': 2.638939625494068e-09, 'W2': 6.219064326931303e-09, 'b2': 1.392913356129677e-07}
#再怎么随，他的结果都是1e-5~1e-10，affine层的梯度级别就不对，如果我换成他的affine，看一下梯度数量级。



