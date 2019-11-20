from collections import OrderedDict


#说OrderedDict可以按顺序，其实普通dict也是按顺序啊，有反例么？？？？
# 不同顺序同样键值对的dict是可以==的，而OrderedDict不能，问题是，这和神经网络layers用这个结构，有什么必然联系？

dx = OrderedDict(b=5,c=2,a=7)

print('dx:',dx)

d = OrderedDict()
d['Python'] = 89
d['Swift'] = 92
d['Kotlin'] = 97
d['Go'] = 87

for k,v in d.items():
    print(k,v)

#比不出来，关键是普通dict也是这样的结果
dxx = dict(b=5,c=2,a=7)
print('dxx:',dxx)
for k,v in dxx.items():
    print(k,v)

o1 = OrderedDict(b=5,c=2,a=7)
o2 = OrderedDict(b=5,c=2,a=7)
o3 = OrderedDict(b=5,a=7,c=2)
print(o1==o2)
print(o1==o3)

#区别在这，是说dict之间对比，顺序影响'=='判定结果，但是和神经网络的层使用OrderedDict有什么必然联系？？？？
d1 = dict(b=5,c=2,a=7)
d2 = dict(b=5,c=2,a=7)
d3 = dict(b=5,a=7,c=2)
print(d1==d2)
print(d1==d3)
#但是即便dict无序，for循环迭代遍历还是不影响啊
for k,v in d1.items():
    print(k,v,end=',')
for k,v in d2.items():
    print(k,v,end=',')
for k,v in d3.items():
    print(k,v,end=',')
print()

print('d1:',d1.items())
print('d2:',d2.items())
print('d3:',d3.items())


my_data = {'Python':20,'Swift':32,'Kotlin':43,'Go':25}
d1 = OrderedDict(sorted(my_data.items(),key=lambda t:t[0]))#key
d2 = OrderedDict(sorted(my_data.items(),key=lambda t:t[1]))#value
print('d1:',d1)
print('d2:',d2)
print(d1 == d2)
print(id(d1),id(d2))
d1 = dict(sorted(my_data.items(),key=lambda t:t[0]))#key
d2 = dict(sorted(my_data.items(),key=lambda t:t[1]))#value
print('d1:',d1)
print('d2:',d2)
print(d1 == d2)
print(id(d1),id(d2))
#无论如何，这还是宏观的dict对象的区别，没有说dict内部key-value的区别
#所以说，到底有什么用？为了让不同网络不会存到同一个dict？这俩dict的id本来就不同，不可能串啊
#那还了得？两个长一样的容器，存期东西来就共享了？
#至于另一个用处，popitem，貌似在网络层这也用不上


d = OrderedDict.fromkeys('abcde')
print(d)
d.move_to_end('b')
print(d.keys())
d.move_to_end('b',last=False)
print(d.keys())
print(d.popitem()[0])#[0]is key,while[1] is value
print(d)
print(d.popitem(last=False)[0])
print(d)





print('regular dictionary')
d = {}
d['a'] = 'A'
d['b'] = 'B'
d['c'] = 'C'

for k,v in d.items():
    print(k,v)
print('order dictionary')
d1 = OrderedDict()
d1['a'] = 'A'
d1['b'] = 'B'
d1['c'] = 'C'
d1['1'] = '1'
d1['2'] = '2'
for k,v in d1.items():
    print(k,v)


#那么添加重复元素会怎样？其实就是覆盖操作
# 同时测试，如果我后改b，算覆盖原来b，还是新添加,都一样！！！
d1 = OrderedDict()
d1['a'] = 'A'
d1['b'] = 'B'
d1['c'] = 'C'
d1['b'] = 'A'
print(d1)
for k,v in d1.items():
    print(k,v)
d1 = dict()
d1['a'] = 'A'
d1['b'] = 'B'
d1['c'] = 'C'
d1['b'] = 'A'
print(d1)
for k,v in d1.items():
    print(k,v)



# dict1 = OrderedDict()
# dict2 = {}
#
# dict1['d'] = 4
# dict1['b'] = 2
# dict1['a'] = 100
# dict1['c'] = 3
# del (dict1['a'])
# dict1['a'] = 101
#
# dict2['d'] = 4
# dict2['b'] = 2
# dict2['a'] = 100
# dict2['c'] = 3
# del (dict2['a'])
# dict2['a'] = 101
#
# for k in dict1:
#     print(dict1[k], )
# for k in dict2:
#     print(dict2[k])
# # for v in dict1.values():
# #     print(v)
# # for v in dict2.values():
# #     print(v)