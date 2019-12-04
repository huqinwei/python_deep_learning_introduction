from chap05_multi_layer import MultiLayer
from chap05_add_layer import AddLayer

#回看了他的参考代码，其实没有直接写，是用的封装的层，可见，深度网络不封装根本没法写。

if __name__ == '__main__':
    apple_price = 100
    apple_num = 2
    apple_cost_layer = MultiLayer()
    apple_cost = apple_cost_layer.forward(apple_price,apple_num)#左右顺序一定要记住，backward要靠自己去保持一致,加法无所谓，乘法一定要

    orange_price = 150
    orange_num = 3
    orange_cost_layer = MultiLayer()
    orange_cost = orange_cost_layer.forward(orange_price,orange_num)

    cost_layer = AddLayer()
    cost = cost_layer.forward(apple_cost,orange_cost)

    tax = 1.1
    final_cost_layer = MultiLayer()
    final_cost = final_cost_layer.forward(cost,tax)

    print('final cost:',final_cost)

    dfinal_cost = 1.0
    dcost,dtax = final_cost_layer.backward(dfinal_cost)

    dapple_cost,dorange_cost = cost_layer.backward(dcost)

    dapple_price,dapple_num = apple_cost_layer.backward(dapple_cost)#保持一致性
    dorange_price,dorange_num = orange_cost_layer.backward(dorange_cost)

    print(dapple_num,dapple_price)
    print(dorange_num,dorange_price)
    print('dtax',dtax)



