class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum#用来做全局平均的衰减
        self.input_shape = None  # Conv层的情况下为4维，全连接层的情况下为2维

        # 测试时使用的全局平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var

        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):#封装：修改卷积模式到FC模式，
        # BN应该是有一个channel一个平均值的模式？这里channel被平铺了，每个feature“像素点”一个平均值。
        # 所以这可能不算最好的conv实例，权当是个FC罢,就是遇到Conv强行转FC的一个形式
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape#用了channel_first
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)#恢复原样

    def __forward(self, x, train_flg):
        if self.running_mean is None:#全局平均值的初始化，作者的参考代码全都是根据第一次前向传播确定形状的模式
            N, D = x.shape
            # 这里也可以看出，是根据每个feature一个平均值，所以好像ConvNet不友好？但是BN层又不该改变维度，做了平均值最终也是要恢复的，注意，主要是平均值数值本身的不同，一个channel一个平均值理论上更平均一些罢了。
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:#训练时（在纸上做公式的分解）
            mu = x.mean(axis=0)#计算N个xi的平均值，N坍塌，D维持不变
            xc = x - mu#x零中心化（N,D）
            var = np.mean(xc ** 2, axis=0)#分子：计算标准差（简化了，因为xc=x-mu是零中心，所以xi-mu就成了xc，然后np.mean）
            std = np.sqrt(var + 10e-7)#分母，标准差，防零
            xn = xc / std#这是标准化后的输出，恢复分布的操作out= self.gamma*xn + self.beta提到了if外边，为了和else共享代码

            self.batch_size = x.shape[0]
            self.xc = xc#这些都被记录了下来，为了反向传播方便使用
            self.xn = xn
            self.std = std
            # 这两个是记录全局平均，我也好奇全局平均怎么记录，总不能一直累加，最后测试时记住数量除一次？更不能每次简单对冲进去？因为batch不一样！
            # 就算除以batch再去更新，那么权重也不太对，之前batch=10000被算成1份，现在batch=10也被算成一份，就冲淡了10000份？所以看到，他是用了momentum来做滑动平均的更新。
            # 这些细节不去实现真的很不容易想到！！！！
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:#这个简单：测试时候直接用全局平均值套公式计算一下就行了
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta#共享操作：恢复分布
        return out

    def backward(self, dout):#同样的，封装了一个ConvNet维度处理的过程
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        #beta和gamma是最外层的两个操作，反向传播最简单，由xn*gamma+beta=out可知
        dbeta = dout.sum(axis=0)#细节：x的坍缩，一个batch合并成一个值。
        dgamma = np.sum(self.xn * dout, axis=0)#同上
        #下面的不墨迹了，其实就是前向传播的过程，逐步反推回去
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size#用来传递给前一层

        self.dgamma = dgamma#本层需要的结果，保存，用于更新参数
        self.dbeta = dbeta#本层需要的结果，保存，用于更新参数

        return dx
