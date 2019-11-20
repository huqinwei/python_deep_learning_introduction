import numpy as np

def numerical_diff_v1(f,x):
    h = 1e-4#1e-50 useless,because of underflow
    # debug_a = f(x+h)
    # debug_b = f(x)
    return (f(x+h)-f(x)) / h
def numerical_diff_v1_2(f,x):#v1 is not fair
    h = 1e-4#-50 useless
    # debug_a = f(x+h)
    # debug_b = f(x)
    # return (f(x+2*h)-f(x)) / 2*h#wrong expression,this will multiply by h
    return (f(x+2*h)-f(x)) / (2*h)
def numerical_diff(f,x):#new
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


def func(x):
    y = 2*x
    return y
def func2(x):
    y = x**2
    return y
def func3(x):
    # print(0.01*x**2)
    return 0.01*x**2 + 0.1*x

if __name__ == '__main__':

    x = 5
    analytic_diff = 2
    analytic_diff2 = 2*x
    analytic_diff3 = 0.01*x + 0.1
    print('func3:',func3(1))
    print('func3:',func3(2))

    print('func3 diff:',numerical_diff(func3,x,))#1.9999999999908982e-09   0.1999999999990898
    print('func3 diff:',numerical_diff_v1(func3,x,))#func3 diff: 0.20000099999917254
    print('func3 diff:',numerical_diff_v1_2(func3,x,))




    print('################################################\nfunc2:')
    diff_v1 = numerical_diff_v1(func2,x,)
    diff = numerical_diff(func2,x,)
    diff_v1_v2 = numerical_diff_v1_2(func2,x,)
    print('diff:'.rjust(10),diff)
    print('diff_v1:'.rjust(10),diff_v1)
    print('diff_v1_2:'.rjust(10),diff_v1_v2)
    if np.abs(diff - analytic_diff2) < np.abs(diff_v1 - analytic_diff2):
        print('the new diff are more accuracy than diff_v1!')

    elif np.abs(diff - analytic_diff2) == np.abs(diff_v1 - analytic_diff2):
        print('draw')
    else:
        print('ah')

    if np.abs(diff - analytic_diff2) < np.abs(diff_v1_v2 - analytic_diff2):
        print('the new diff are more accuracy than diff_v1_v2!')

    elif np.abs(diff - analytic_diff2) == np.abs(diff_v1_v2 - analytic_diff2):
        print('draw')
    else:
        print('ah')

    print('################################################\nfunc:')

    diff_v1 = numerical_diff_v1(func,x,)
    diff = numerical_diff(func,x,)
    diff_v1_v2 = numerical_diff_v1_2(func,x,)

    print('diff:'.rjust(10),diff)
    print('diff_v1:'.rjust(10),diff_v1)
    print('diff_v1_2:'.rjust(10),diff_v1_v2)
    if np.abs(diff - analytic_diff) < np.abs(diff_v1 - analytic_diff):
        print('the new diff are more accuracy than diff_v1!')
    elif np.abs(diff - analytic_diff) == np.abs(diff_v1 - analytic_diff):
        print('draw')
    else:
        print('ah')


    if np.abs(diff - analytic_diff) < np.abs(diff_v1_v2 - analytic_diff):
        print('the new diff are more accuracy than diff_v1_2!')
    elif np.abs(diff - analytic_diff) == np.abs(diff_v1_v2 - analytic_diff):
        print('draw!!!!!!!!!!!!!!!!!!!!')
    else:
        print('ah,reverse!!!!!!!!!!!!!!!!')
    # print('np.abs(diff - analytic_diff) < np.abs(diff_v1 - analytic_diff):',np.abs(diff - analytic_diff),np.abs(diff_v1 - analytic_diff),np.abs(diff - analytic_diff) < np.abs(diff_v1 - analytic_diff))



    print('################################################\nfunc3:')
    diff_v1 = numerical_diff_v1(func3,x,)
    diff = numerical_diff(func3,x,)
    diff_v1_v2 = numerical_diff_v1_2(func3,x,)
    print('diff:'.rjust(10),diff)
    print('diff_v1:'.rjust(10),diff_v1)
    print('diff_v1_2:'.rjust(10),diff_v1_v2)
    if np.abs(diff - analytic_diff3) < np.abs(diff_v1 - analytic_diff3):
        print('the new diff are more accuracy than diff_v1!')
    elif np.abs(diff - analytic_diff3) == np.abs(diff_v1 - analytic_diff3):
        print('draw')
    else:
        print('ah')
    # print('np.abs(diff - analytic_diff3) * 100 < np.abs(diff_v1 - analytic_diff3):',np.abs(diff - analytic_diff3) * 100 < np.abs(diff_v1 - analytic_diff3))


    if np.abs(diff - analytic_diff3) < np.abs(diff_v1_v2 - analytic_diff3):
        print('the new diff are more accuracy than diff_v1_2!')
    elif np.abs(diff - analytic_diff3) == np.abs(diff_v1_v2 - analytic_diff3):
        print('draw')
    else:
        print('ah')







#maybe multiply by 10 is a error,makes no sense
# print(10e-50)
# print(1e-49)
# print(10e-50 == 1e-49)
# print(np.float32(10e-50))
# print(np.float64(10e-50))
# test = 10e-50
# print(type(test))