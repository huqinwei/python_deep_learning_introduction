import numpy as np

def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    res = np.sum(w*x)+b
    if res > 0:#default rules:res > 0,output = 1,这个写法更符合感知器和激活阈值的客观意义
        return 1
    else:
        return 0
if __name__ == '__main__':
    print(NAND(1,1))
    print(NAND(1,0))
    print(NAND(0,0))
