# coding=utf-8

'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

@author: pbharrin

k-近邻算法
k-近邻算法采用测量不同特征值之间的距离方法进行分类

k-近邻算法的优缺点
优点 精度高 对异常值不敏感 无数据输入假定
缺点 计算复杂度高 空间复杂度高
适用数据范围： 数值型和标称型

标称型目标变量的结果只在有限目标集中取值，如真与假、动物分类集合{ 爬行类、鱼类、哺乳类、两栖类} ；数值型目标变量则可以从无限的数值集合中取值，如0.100、42.001、1000.743 等。

kNN算法的工作原理：
存在一个样本数据集合，也称作训练样本集，并且样本集中每个数据都存在标签。即我们知道样本集中每一数据与所属分类的对应关系
输入没有标签的新数据后，将新数据的每个特征和样本集中数据对应的特征进行比较，然后算法提取样本集中特征最相似的数据(最近邻)
和标签分类。一般来说，我们只选择样本数据集中前k个最相似的数据，这就是k-近邻算法的出处，通常k是不大于20的整数。最后，选择k
个最相似数据中出现次数最多的分类，作为新数据的分类




                  k-近邻算法的一般流程
(1)收集数据:可以使用任何方法。
(2)准备数据:距离计算所需要的数值，最好是结构化的数据格式。
(3)分析数据:可以使用任何方法。
(4)训练算法:此步骤不适用于k-近部算法。
(5)测试算法:计算错误率。
(6)使用算法:首先需要输入样本数据和结构化的输出结果，然后运行k-近邻算法判定输
  入数据分别属于哪个分类，最后应用对计算出的分类执行后续的处理。



从文本文件中解析数据
伪代码如下：
1、计算已知类别数据集中的点与当前点之间的距离
2、按照距离递增次序排列
3、选取与当前点距离最小的k个点
4、确定前k个点所在类别的出现频率
5、返回前k个点出现频率最高的类别作为当前点的预测分类


k-近邻算法是分类数据最简单有效的算法 k-近邻算法基于实例的学习，使用算法时，必须有接近实际数据的训练样本数据
k-近邻算法必须保存全部数据集，这样训练数据集很大的话，必须使用大量的存储空间。由于必须对数据集中每个数据计算距离值，实际使用时可能非常耗时
k-近邻算法的另一个缺陷是无法给出任何数据的基础结构信息，因此无法知晓平均实例样本和典型实例样本具有什么特征

numpy科学计算包
运算符模块
'''

from numpy import *
import operator
from os import listdir

'''距离的计算
classify0函数有4个输入参数：
用于分类的输入向量inX，输入的训练样本集为dataSet，标签向量labels，最后的参数k表示用于选择最近邻居的数目
其中标签向量的元素数目和矩阵dataSet的行数相同，使用欧氏距离公式，计算两个想亮点xA和xB之间的距离

计算两个向量点xA xB之间的距离
欧氏距离公式：
d=sqrt((xA0-xB0)^2+(xA1-xB1)^2)

计算完所有点之间的距离后，可以对数据按照从小到大的次序排列。然后，确定前K个距离最小元素所在的主要分类，输入K总是正整数；最后
将classCount()字典分解为元祖列表，然后使用程序第二行导数运算符模块的itermgetter方法，按照第二个元素
的次序对元祖进行排序。此处的排序为逆序，即按照从最大到最小次序排序，最后返回发生频率最好的元素标签
'''


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 数组的大小
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 函数的形式是tile(A,reps)，参看博客
    sqDiffMat = diffMat ** 2  # **平方的意思
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5  # 开平方
    # 按照距离递增次序排列  计算完所有点之间的距离后，可以对数据按照从小到大的次序进行排序，然后确定前k个距离最小元素所在的主要分类，输入k总是正整数；最后，将classCount字典分解为元祖列表，然后使用程序第二行导入运算符模块的itemgetter方法，按照第二个元素的次序对元祖进行排序
    sortedDistIndicies = distances.argsort()

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


'''
有四组数，魅族数据有两个已知的属性或特征值，group矩阵每行包含一个不同的数据，可以把它想象成某个日志文件中
不同的测量点或者入口。因为人脑的限制，通常只能可视化处理三维以下的事务。因此为了实现数据可视化，对于每个
数据点通常只使用两个特征。
向量label包含每个数据点的标签信息，label包含的元素个数等于group矩阵行数
这里（1.0,1.1）定义为A (0,0.1)定义为B
'''


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


'''
在约会网站上使用k-近邻算法
1、收集数据：提供样本文件
2、准备数据：使用Python解析文本文件
3、分析数据：使用matplotlib画二维扩散图
4、训练算法：此步骤不适合k-近邻算法
5、测试算法：测试样本和非测试样本区别在于：测试样本已经完成分类的数据，如果预测分类与实际类别不同，则标为error
6、使用算法：产生简单的命令行程序，然后可以输入一些特征数据以判断对方是否为自己喜欢的类型
'''
# 确保样本文件和py文件在同一目录下，样本数据存放在datingTestSet.txt文件中
'''
样本主要包含了一下内容
1、每年获得的飞行常客里程数
2、玩视频游戏所耗时间百分比
3、每周消费的冰激凌公升数

>>> import matplotlib
>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111)
>>> ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
<matplotlib.collections.PathCollection object at 0x03EF6690>
>>> plt.show()
由于没有使用样本分类的特征值，很难看到任何有用的数据模式信息，一般来说
采用色彩或者其他记号来标记不同样本分类，以便更好地理解数据信息
>>> ax.scatter(datingDataMat[:,1],datingDataMat[:,2,15*array(datingLabels),15*datingLabels])  暂时有误，需要解决
利用颜色以及尺寸标识了数据点的属性类别，带有养病呢分类标签的约会数据散点图，虽然能够比较容易的区分数据点从属类别，但依然很难根据这张图给出结论性的信息

'''


def file2matrix(filename):
    fr = open(filename)
    f_lines = fr.readlines()
    numberOfLines = len(f_lines)  # get the number of lines in the file 得到文件的行数

    returnMat = zeros((numberOfLines,
                       3))  # prepare matrix to return  创建以0填充的矩阵numpy，为了简化处理，将该矩阵的另一维度设置为固定值3，可以根据自己的需求增加相应的代码以适应变化的输入值
    classLabelVector = []  # prepare labels return
    # fr = open(filename)

    index = 0
    for line in f_lines:  # 循环处理文件中的每行数据，首先使用line.strip截取掉所有的回车字符，然后使用tab字符\t将上一步得到的整行数据分割成一个元素列表
        line = line.strip()

        listFromLine = line.split('\t')

        returnMat[index, :] = listFromLine[0:3]  # 选取前3个元素，将其存储到特征矩阵中
        classLabelVector.append(listFromLine[-1])  # Python语言可以使用索引值-1表示列表中的最后一列元素，利用这种负索引，可以将列表的最后一列存储到向量classLabelVector中。注意：必须明确的通知解释器，告诉它列表中存储的元素值为整形，否则Python语言会将这些元素当做字符串来处理  listFromLine前不能加int否则报错
        index += 1
    return returnMat, classLabelVector


'''
归一化数值
多种特征同等重要时(等权重)，处理不同取值范围的特征值时，通常采用数值归一化，将取值范围处理为0~1或者-1~1之间
newValue = {oldValue-min}/(max-min)
min和max分别是数据及数据集中的最小特征值和最大特征值。虽然改变数值取值范围增加了分类器的复杂度，但为了得到精确结果，必须这样做
autoNorm将数字特征值转换为0~1
>>> reload(kNN)
<module 'kNN' from 'C:\Users\kernel\Documents\python\kNN.py'>
>>> normMat,ranges,minVals = kNN.autoNorm(datingDataMat)

函数autoNorm()中，将每列的最小值放在变量minValue中，将最大值放在变量maxValue中。其中
dataSet.min(0)中的参数0使得函数可以从列中选取最小值，而不是选取当前行的最小值。然后，
函数计算可能的取值范围，并创建新的矩阵。

为了归一化特征值，必须使用当前值减去最小值，除以取值范围。需要注意的是：特征值矩阵有1000*3
个值，而minVals和range的值都为1*3.使用Numpy库中tile()函数将变量内容复制成输入矩阵大
小的矩阵，具体特征值相除，而对于某些数值处理软件包，/可能意味着矩阵除法在Numpy同样库中，
矩阵除法需要使用函数linalg.solve(matA,matB)
'''

def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 每列的最小值  参数0可以从列中选取最小值而不是选取当前行的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals  # 函数计算可能的取值范围，并创建新的返回矩阵，为了归一化特征值，必须使用当前值减去最小值，然后除以取值范围
    normDataSet = zeros(
        shape(dataSet))  # 注意事项：特征值矩阵有1000*3个值。而minVals和range的值都为1*3.为了解决这个问题使用numpy中tile函数将变量内容复制成输入矩阵同样大小的矩阵
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals


'''
对于分类器而言错误率就是分类器给出错误结果的次数除以测试数据的总数，完美分类器错误率为0，错误率为1的分类器不会给出任何正确的分类结果
在代码中设定一个计数器变量，每次分类器错误的分类数据，计数器就+1，程序执行完成后计算器的结果除以数据点总数即为错误率
>>> kNN.datingClassTest()
NameError: global name 'datingDataMat' is not defined  悬而未决
'''


def datingClassTest():
    hoRatio = 0 #10

    datingDataMat, datingLables = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print(        "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print(    "the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(    errorCount)


'''
该方法有问题需要改正 (已作更正)

约会网站预测函数
'''


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input( \
        "percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = int(classify0((inArr - \
                                      minVals) / ranges, normMat, datingLabels, 3))
    print(    "You will probably like this person:", \
    resultList[classifierResult - 1])


'''
手写识别系统
构造的系统只能识别数字0~9，需要是别的数字已经使用图像处理软件，处理成具有相同的色彩和大小：
宽高是32*32的黑白图像
1、收集数据 提供文本文件
2、准备数据 编写函数classify0(),将图像格式转换成分类器使用的list格式
3、分析数据 在Python命令提示符中检查数据，确保它符合要求
4、训练算法 此步骤不适合k-近邻算法
5、测试算法 测试样本和非测试样本区别在于：测试样本已经完成分类的数据，如果预测分类与实际类别不同，则标为error
6、使用算法 未实现
'''


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


'''

手写数字识别系统的测试代码

testDigits目录中的文件内容存储在列表中，然后可以得到目录中有多少文件，便将其存储到变量m中
创建一个m*1024的训练矩阵，该矩阵的每行数据存储一个图像，可以从文件名中解析出分类数字
该目录下的文件按照规则命名，如文件9_45.txt的分类是9，它是数字9的第45个实例
将类代码存储在hwLabels向量中，使用img2vector载入图像
对testDigits目录中的文件执行相似的操作，不同之处在于我们并不将这个目录下的文件载入矩阵中
而是利用classify0()函数测试该目录下每个文件，由于文件中的值已在0~1之间，所以不需要autoNorm()函数
该算法执行效率不高，因为算法需要为每个测试向量做2000词距离计算，每个距离计算包括了1024个维度浮点计算，总计执行900次
此外还需要为向量准备2M的存储空间  k决策树是k-近邻算法的改进版


'''


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print(        "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print(    "\nthe total number of errors is: %d" % errorCount)
    print(  "\nthe total error rate is: %f" % (errorCount / float(mTest)))


'''
import numpy as np
import operator


# K近邻算法分类爱情片和动作片
# 通过打斗镜头次数和接吻镜头次数区分


# 模拟数据 打斗次数 接吻次数
def init_data():
    data_X = np.array([[0.7,0.7],[0.7,0.8],[0.7,0.9],[0.7,1],[0.7,1.1],[0.7,1.2],[0.7,1.4]])
    data_Y = ['AWB','AWB','AWB','AWB_lsc','LSC','LSC','LSC']
    return data_X,data_Y


# data: 测试数据
# testData: 数据集合
# output:测试数据输出
# k:取最接近数据的前几个
def kNN(data, testData, output, k):
    # 获取测试数据数量
    dataInputRow = testData.shape[0]
    # np.tile 数组沿各个方向复制
    # 这里将输入数据copy和测试数据数量保持一致，用来计算和测试数据的欧式距离
    reduceData = np.tile(data, (dataInputRow,1)) - testData
    squareData = reduceData ** 2
    squareDataSum = squareData.sum(axis = 1)
    distance = squareDataSum ** .5
    sortDistance = distance.argsort()
    print(sortDistance, distance,squareDataSum)
    dataCount = {}
    # 统计排名靠前k数据的爱情片和动作片次数，取次数最高的做为输出
    for i in range(k):
        print("aaa", i, output[sortDistance[i]])
        output_ = output[sortDistance[i]]
        dataCount[output_] = dataCount.get(output_,0) + 1
    sortDataCount = sorted(dataCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortDataCount[0][0]


if __name__ == '__main__':
    data_X,data_Y = init_data()
    print(kNN([0.7,1.4], data_X, data_Y, 3))
'''