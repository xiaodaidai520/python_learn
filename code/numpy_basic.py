#!/usr/bin/env python
# encoding: utf-8

# 导引numpy包
import numpy as np


def print_info(val):
    print("#" * 50 + val + "#" * 50)
    return


if __name__ == '__main__':
    # 1、打印包的版本号
    print(f"当前np的的版本号:{np.__version__}")
    print(f"np的配置信息:{np.show_config()}")

    print("#" * 50, "2、创建空向量", "#" * 50)
    # 2、创建空向量, ctrl + 鼠标左击查看函数使用方法
    print(f"生成一个1维长度为10的0向量:\n {np.zeros(shape=10,dtype=np.int8,order='F')}")
    print(f"生成一个2维长度为10的0向量:\n {np.zeros((2,10),int)}")

    # 3、查看数组的内存大小
    print("#" * 50, "3、查看数组的内存大小", "#" * 50)
    d = np.zeros(2, dtype=np.int8, order='C')
    # order ={'C','F'}  按行和列的顺序存储在内存中(row- or column-wise)
    print(f"向量{d} 的元素个数:{d.size}")
    print(f"向量 {d} 的内存大小{d.itemsize}")

    # 4、获取方法的说明信息
    print_info("4、获取方法的说明信息")
    print("获取 np.zeros 的说明信息:\n", np.info(np.zeros))

    print_info("5、创建一个长度为10并且除了第五个值为1的空向量")

    a = np.zeros(10)
    a[4] = 1
    print(a)
    print_info("6、创建一个值域范围为10,49的向量")
    print(np.arange(10, 50, step=1))
    print_info("7、 反转一个向量")
    a = np.arange(1, 10)
    print(a[::-1])
    print_info("8、3x3 并且值从0到8的矩阵")
    print(np.arange(0, 9).reshape(3, 3))

    print_info("9、到数组[1,2,0,0,4,0]中非0元素的位置索引")
    print(f"数组[1,2,0,0,4,0]中非0元素的位置索引 {np.nonzero([1,2,0,0,4,0])}")
    print_info("10、 3x3 的单位矩阵")
    print(np.eye(3))

    print_info("11、 3x3x3的随机数组")
    print(np.random.random((3, 3, 3)))

    print_info("12、 10x10 的随机数组并找到它的最大值和最小值")
    a = np.random.random((10, 10))
    print(f"最大值{a.max()} 最小值{a.min()}")
    print_info("13、长度为30的随机向量并找到它的平均值 ")
    a = np.random.random(30)
    print(f"平均值 {a.mean()}")

    print_info("14、一个二维数组，其中边界值为1，其余值为0")
    a = np.ones((10, 10))
    a[1:-1, 1:-1] = 0
    print(a)

    print_info("15、存在在数组，如何添加一个用0填充的边界")
    print(np.pad(a, pad_width=1, mode="constant", constant_values=0))
    print_info("16、下面表达达式运行的结果分别是什么?")
    print("""
    0 * np.nan
    np.nan == np.nan
    np.inf > np.nan
    np.nan - np.nan
    0.3 == 3 * 0.1
    """)
    print(f"0 * np.nan --> {0 * np.nan}")
    print(f"np.nan == np.nan --> {np.nan == np.nan}")
    print(f"np.inf > np.nan --> {np.inf > np.nan}")
    print(f"np.nan - np.nan --> {np.nan - np.nan}")
    print(f"0.3 == 3 * 0.1 --> {0.3 == 3 * 0.1}")

    print_info("17、 5x5的矩阵，并设置值1,2,3,4落在其对角线下方位置")
    print(np.diag(1 + np.arange(4), k=-1))
    print_info("18、创建一个8x8 的矩阵，并且设置成棋盘样式")
    a = np.zeros((8, 8), dtype=int)
    a[1::2, ::2] = 1
    a[::2, 1::2] = 1
    print(a)

    print_info("19、一个 (6,7,8) 形状的数组，其第100个元素的索引(x,y,z)是什么?")
    print(np.unravel_index(100, (6, 7, 8)))

    print_info("20、 用tile函数去创建一个 8x8的棋盘样式矩阵")
    print(np.tile(np.array([[0, 1], [1, 0]]), (4, 4)))

    print_info("21、x5的随机矩阵做归一化")
    a = np.random.random((5, 5))
    a_min, a_max = a.min(), a.max()
    print((a - a_min) / a_max - a_min)

    print_info("22、 创建一个将颜色描述为(RGBA)四个无符号字节的自定义dtype")
    color = np.dtype([("r", np.ubyte, 1),
                      ("g", np.ubyte, 1),
                      ("b", np.ubyte, 1),
                      ("a", np.ubyte, 1)])
    print(color)

    print_info("23 、一个5x3的矩阵与一个3x2的矩阵相乘，实矩阵乘积是什么")
    print(np.dot(np.ones((5, 3)), np.ones((3, 2))))

    print_info("24、定一个一维数组，对其在3到8之间的所有元素取反")
    Z = np.arange(11)
    Z[(3 < Z) & (Z <= 8)] *= -1
    print(Z)

    print_info("25、下面脚本运行后的结果是什么?")
    print(f"sum(range(5),-1) --> {sum(range(5),-1)}")
    # from numpy import *
    print(f"np.sum(range(5),-1) --> {np.sum(range(5),-1)}")

    print_info("26、 考虑一个整数向量Z,下列表达合法的是哪个? ")
    try:
        print(f" Z ** Z --> {Z ** Z}")
    except Exception as e:
        print(e.args)
    try:
        print(f" 2 << Z >> 2 --> {2 << Z >> 2}")
    except Exception as e:
        print(e.args)

    try:
        print(f" Z <- Z --> {Z <- Z}")
    except Exception as e:
        print(e.args)

    try:
        print(f" 1j*Z --> {1j*Z}")
    except Exception as e:
        print(e.args)

    try:
        print(f" Z/1/1 --> {Z/1/1}")
    except Exception as e:
        print(e.args)

    try:
        print(f" Z<Z>Z --> {Z<Z>Z}")
    except Exception as e:
        print(e.args)

    print_info("27、下列表达式的结果分别是什么?")
    print(f"np.array(0) / np.array(0) -- > {np.array(0) / np.array(0)}")
    print(f"np.array(0) // np.array(0) -- > {np.array(0) // np.array(0)}")
    print(f"np.array([np.nan]).astype(int).astype(float) -- > {np.array([np.nan]).astype(int).astype(float)}")

    print_info("28、如何从零位对浮点数组做舍入 ")
    Z = np.random.uniform(-10, +10, 10)
    print(np.copysign(np.ceil(np.abs(Z)), Z))

    print_info("29、如何找到两个数组中的共同元素?")
    Z1 = np.random.randint(0, 10, 10)
    Z2 = np.random.randint(0, 10, 10)
    print(f"Z1 --> {Z1}")
    print(f"Z2 --> {Z2}")
    print(np.intersect1d(Z1, Z2))

    print_info("30、如何忽略所有的 numpy 警告(尽管不建议这么做)? ")
    defaults = np.seterr(all="ignore")
    Z = np.ones(1) / 0

    _ = np.seterr(**defaults)
    with np.errstate(divide='ignore'):
        Z = np.ones(1) / 0
        print(Z)

    print_info("31、下面的表达式是正确的吗?")
    print(f"np.sqrt(-1) --> {np.sqrt(-1)}")
    print(f'np.emath.sqrt(-1) --> {np.emath.sqrt(-1)}')
    print(f"np.sqrt(-1) == np.emath.sqrt(-1) --> {np.sqrt(-1) == np.emath.sqrt(-1)}")

    print_info("32、得到昨天，今天，明天的日期?")
    yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
    today = np.datetime64('today', 'D')
    tomorrow = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
    print("Yesterday is " + str(yesterday))
    print("Today is " + str(today))
    print("Tomorrow is " + str(tomorrow))
    print_info("33、得到所有与2016年7月对应的日期")
    Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
    print(Z)
    print_info("34、如何直接在位计算(A+B)\*(-A/2)(不建立副本)?")
    A = np.ones(3) * 1
    B = np.ones(3) * 2
    C = np.ones(3) * 3
    print(f"np.add(A,B,out=B) --> {np.add(A,B,out=B)}")
    print(f"np.divide(A,2,out=A) --> {np.divide(A,2,out=A)}")
    print(f"np.negative(A,out=A) --> {np.negative(A,out=A)}")
    print(f"np.multiply(A,B,out=A) --> {np.multiply(A,B,out=A)}")

    print_info("35、用五种不同的方法去提取一个随机数组的整数部分")
    Z = np.random.uniform(0, 10, 10)
    print(Z - Z % 1)
    print(np.floor(Z))
    print(np.ceil(Z) - 1)
    print(Z.astype(int))
    print(np.trunc(Z))
    print("36、创建一个5x5的矩阵，其中每行的数值范围从0到4")
    Z = np.zeros((5, 5))
    Z += np.arange(5)
    print(Z)
    print_info("37、通过考虑一个可生成10个整数的函数，来构建一个数组")


    def generate():
        for x in range(10):
            yield x


    Z = np.fromiter(generate(), dtype=float, count=-1)
    print(Z)

    print_info("38、创建一个长度为10的随机向量，其值域范围从0到1，但是不包括0和1 ")
    print(np.linspace(0, 1, 11, endpoint=False)[1:])
    print_info("39 、 创建一个长度为10的随机向量，并将其排序")
    Z = np.random.random(10)
    Z.sort()
    print(Z)

    print_info("40、对于一个小数组，如何用比 np.sum更快的方式对其求和")
    Z = np.arange(10)
    np.add.reduce(Z)
    print(Z)

    print_info("41、对于两个随机数组A和B，检查它们是否相等")
    A = np.random.randint(0, 2, 5)
    B = np.random.randint(0, 2, 5)
    # Assuming identical shape of the arrays and a tolerance for the comparison of values
    equal = np.allclose(A, B)
    print(f"allclose --> {equal}")

    # 方法2
    # Checking both the shape and the element values, no tolerance (values have to be exactly equal)
    equal = np.array_equal(A, B)
    print(f"array_equal --> {equal}")

    print_info("42、创建一个只读数组(read-only) ")
    try:
        Z = np.zeros(10)
        Z.flags.writeable = False
        Z[0] = 1
    except Exception as e:
        print(f" only read {e}")

    print_info("43. 将笛卡尔坐标下的一个10x2的矩阵转换为极坐标形式")
    Z = np.random.random((10, 2))
    X, Y = Z[:, 0], Z[:, 1]
    R = np.sqrt(X ** 2 + Y ** 2)
    T = np.arctan2(Y, X)
    print(R)
    print(T)

    print_info("44 、创建一个长度为10的向量，并将向量中最大值替换为1")
    Z = np.random.random(10)
    Z[Z.argmax()] = 0
    print(Z)

    print_info("45 、 创建一个结构化数组，并实现 x 和 y 坐标覆盖 [0,1]x[0,1] 区域")
    Z = np.zeros((5, 5), [('x', float), ('y', float)])
    Z['x'], Z['y'] = np.meshgrid(np.linspace(0, 1, 5),
                                 np.linspace(0, 1, 5))
    print(Z)

    print_info("46、 给定两个数组X和Y，构造Cauchy矩阵C (Cij =1/(xi - yj))")
    X = np.arange(8)
    Y = X + 0.5
    C = 1.0 / np.subtract.outer(X, Y)
    print(np.linalg.det(C))

    print_info("47、打印每个numpy标量类型的最小值和最大值？")
    for dtype in [np.int8, np.int32, np.int64]:
        print(f"{dtype} 最小值 {np.iinfo(dtype).min}")
        print(f"{dtype} 最大值 {np.iinfo(dtype).max}")

    print("分割线")

    for dtype in [np.float32, np.float64]:
        print(f"{dtype} 最小值 {np.finfo(dtype).min}")
        print(f"{dtype} 最大值 {np.finfo(dtype).max}")
        print(f"{dtype} eps值 {np.finfo(dtype).eps}")

    print_info("48、如何打印一个数组中的所有数值?")
    np.set_printoptions(threshold=np.nan)
    Z = np.zeros((16, 16))
    print(Z)

    print_info("49、给定标量时，如何找到数组中最接近标量的值")
    Z = np.arange(100)
    v = np.random.uniform(0, 100)
    index = (np.abs(Z - v)).argmin()
    print(Z[index])

    print_info("50、创建一个表示位置(x,y)和颜色(r,g,b)的结构化数组")
    Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                      ('y', float, 1)]),
                       ('color',    [ ('r', float, 1),
                                      ('g', float, 1),
                                      ('b', float, 1)])])
    print (Z)

    print_info("51、 对一个表示坐标形状为(100,2)的随机向量，找到点与点的距离")
    Z = np.random.random((10,2))
    X,Y = np.atleast_2d(Z[:,0], Z[:,1])
    D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
    print (D)

    # # 方法2
    # Much faster with scipy
    print("----------方法2------------------------------")
    import scipy
    # Thanks Gavin Heverly-Coulson (#issue 1)
    import scipy.spatial
    D = scipy.spatial.distance.cdist(Z,Z)
    print (D)

    print_info("52、如何将32位的浮点数(float)转换为对应的整数(integer)?")
    Z = np.arange(10, dtype=np.int32)
    Z = Z.astype(np.float32, copy=False)
    print (Z)

    print_info("53、如何读取以下文件?")
    """
    1, 2, 3, 4, 5
    6,  ,  , 7, 8
     ,  , 9,10,11
     """





