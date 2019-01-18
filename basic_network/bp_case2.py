import math
import random


# -*- coding:utf-8 -*-
# random.seed(0) #使得随机值可预测

class tools():  # 定义一些工具函数
    def rand(a, b):  # a到b之间一个随机数
        return (b - a) * random.random() + a

    def set_m(n, m):  # 产生一个n*m的矩阵
        a = []
        for i in range(n):
            a.append([0.0] * m)
        return a

    def sigmoid(x):  # 定义sigmoid函数和它的导数
        return 1.0 / (1.0 + math.exp(-x))

    def sigmoid_derivate(x):
        return x * (1 - x)  # sigmoid函数的导数


class bpnn():
    def __init__(self):  # 初始化定义各个参数
        self.inputn = 0
        self.hiddenn = 0
        self.outputn = 0  # 分别初始化输入层，隐层和输出层的神经元个数
        self.outputcells = []
        self.hiddencells = []  # 定义输出和隐层神经元的阈值
        self.iw = []
        self.ow = []  # 输入层权值和输出层权值
        self.iwco = []
        self.owco = []  # 隐层和输出层矫正矩阵
        self.i = []
        self.h = []
        self.o = []  # 神经元的值
        self.n = 0.05  # 学习率

    def setup(self, inn, outn, hiddenn):  # 输入参数,建立一个神经网络
        self.inputn = inn;  # 输入层
        self.outputn = outn;  # 输出层
        self.hiddenn = hiddenn;  # 隐层

        # 随机初始化权值和阈值
        self.i = [0.0] * self.inputn
        self.o = [0.0] * self.outputn
        self.h = [0.0] * self.hiddenn

        self.iwc0o = tools.set_m(self.inputn, self.hiddenn)
        self.owco = tools.set_m(self.hiddenn, self.outputn)  # 初始化矫正矩阵备用

        self.hiddencells = [1.0] * self.hiddenn
        self.outputcells = [1.0] * self.outputn
        for i in range(self.hiddenn):  # 随机初始化阈值（0，1）
            self.hiddencells[i] = tools.rand(0, 1)
        for i in range(self.outputn):
            self.outputcells[i] = tools.rand(0, 1)

        self.iw = tools.set_m(self.inputn, self.hiddenn)
        self.ow = tools.set_m(self.hiddenn, self.outputn)
        for i in range(self.inputn):  # 随机初始化权值（0，1）
            for j in range(self.hiddenn):
                self.iw[i][j] = tools.rand(0, 1)
        for i in range(self.hiddenn):  # 随机初始化权值（0，1）
            for j in range(self.outputn):
                self.ow[i][j] = tools.rand(0, 1)

    def predict(self, n):  # 预测输出,n是输入列表
        self.i = n;
        for i in range(self.hiddenn):  # 更新隐层
            self.h[i] = 0.0
            for j in range(self.inputn):
                self.h[i] = self.h[i] + self.i[j] * self.iw[j][i]

            self.h[i] = tools.sigmoid(self.h[i] - self.hiddencells[i])

        for i in range(self.outputn):  # 更新输出层
            self.o[i] = 0.0
            for j in range(self.hiddenn):
                self.o[i] = self.o[i] + self.h[j] * self.ow[j][i]
            # print(self.o,self.outputcells)

            self.o[i] = tools.sigmoid(self.o[i] - self.outputcells[i])
            # print(self.o)

    def update(self, n):  # 更新权值和阈值，n是正确的输出
        g = [0.0] * self.outputn  # 输出层的梯度
        e = [0.0] * self.hiddenn  # 隐层的梯度
        for i in range(self.outputn):  # 计算输出层的梯度
            y = self.o[i]
            g[i] = y * (1 - y) * (n[i] - y)

        for i in range(self.hiddenn):  # 计算隐层的梯度
            wg = 0
            for j in range(self.outputn):
                wg = wg + self.ow[i][j] * g[j]
            # print(self.h[i])
            e[i] = self.h[i] * (1 - self.h[i]) * wg

        for i in range(self.hiddenn):  # 更新隐层到输出层的权值ow
            for j in range(self.outputn):
                self.ow[i][j] = self.ow[i][j] + self.n * g[j] * self.h[i]
        for i in range(self.inputn):  # 更新输入层到隐层的权值iw
            for j in range(self.hiddenn):
                self.iw[i][j] = self.iw[i][j] + self.n * e[j] * self.i[i]

        for i in range(self.hiddenn):  # 更新隐层的阈值
            self.hiddencells[i] = self.hiddencells[i] - self.n * e[i]
        for i in range(self.outputn):  # 更新输出层阈值
            self.outputcells[i] = self.outputcells[i] - self.n * g[i]

    def test(self, tn, tp):  # 测试测试集
        acc = 0
        k = 0
        for i in tn:
            self.predict(i)
            print(self.o, tp[k], )
            if self.o[0] > 0.5 and self.o[1] < 0.5 and tp[k][0] == 1 and tp[k][1] == 0:
                acc = acc + 1
                print(acc, )
            if self.o[0] < 0.5 and self.o[1] > 0.5 and tp[k][0] == 0 and tp[k][1] == 1:
                acc = acc + 1
                print(acc)
            k = k + 1
        acc = acc / k
        print(acc)

    def train(self, n, p, tn, tp, vn, vp):  # n是训练集数据，p是训练集的真实输出,tn是测试集输入，tp是测试集输出，vn，vp是训练集
        self.setup(len(n[0]), len(p[0]), int(math.sqrt(len(n[0]) + len(p[0]))) + 5)  # 新建一个神经

        for time in range(1000):  # 最大训练次数
            acct1 = 0  # 训练集累计误差
            accv1 = 0  # 验证机累计误差
            acct2 = 0  # 训练集累计误差
            accv2 = 0  # 验证机累计误差
            k = 0
            for i in n:
                self.predict(i)
                self.update(p[k])

                ek = 0

                for j in range(self.outputn):
                    ek = ek + math.pow((p[k][j] - self.o[j]), 2)
                ek = ek / 2  # 均方误差
                acct2 = acct2 + ek  # 平均误差
                k = k + 1
            acct2 = acct2 / k
            k = 0
            for i in vn:
                self.predict(i)

                ek = 0
                for j in range(self.outputn):
                    ek = ek + math.pow((vp[k][j] - self.o[j]), 2)
                ek = ek / 2  # 均方误差
                accv2 = accv2 + ek  # 平均误差
                k = k + 1
            accv2 = accv2 / k
            if acct2 > acct1 and accv2 < accv1 and i != 0:  # 早停
                break;
            if acct2 < 0.01:
                break;
            acct1 = acct2
            accv1 = accv2
            # print(acct1)


if __name__ == '__main__':
    nn = bpnn()
    # nn.train([[1,2,3],[4,5,6]],[[1,2],[1,2]],[[4,5,6],[4,1,6]],[[2,1],[1,1]],[[4,5,6],[4,1,6]],[[2,1],[1,1]])
    f = open('D:/毕业设计\数据集1/australian.txt', 'r')
    a = f.readlines()
    b = []
    for i in a:  # 一些对数据的简单的处理 ，可以忽略
        i = i.strip("\n")
        i = i.split(",")
        c = list(i)
        for j in range(len(c)):
            c[j] = float(c[j])
        max1 = max(c[0:len(c) - 1])
        min1 = min(c[0:len(c) - 1])
        for j in range(len(c) - 1):
            c[j] = (c[j] - min1) / (max1 - min1)
        b.append(c)
    n = []
    p = []
    tn = []
    tp = []
    vn = []
    vp = []
    k = 0
    for i in b:  # 一些对数据的简单的处理 ，可以忽略
        p.append(i[len(i) - 1:len(i)])
        if p[k][0] == 0:
            p[k].append(1.0)
        if p[k][0] == 1:
            p[k].append(0.0)
        n.append(i[0:len(i) - 1:1])
        k = k + 1
    tn = n[500:600]
    tp = p[500:600]
    vn = n[600:620]
    vp = p[600:620]
    n = n[0:500]
    p = p[0:500]
    nn.train(n, p, tn, tp, vn, vp)
    nn.test(tn, tp)
    f.close()
