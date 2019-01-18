from numpy import *
import operator

def creatDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify(inX,dataSet,labels,k):
    print("ccc", inX, dataSet, labels, k)
    numSamples = dataSet.shape[0]
    diffMat = tile(inX,(numSamples,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    print("AA:\n", numSamples, diffMat, sqDiffMat, "\nsss", sqDistances, distances, sortedDistIndicies)
    classCount = {}
    for i in range(k):
        voteILabel = labels[sortedDistIndicies[i]]
        classCount[voteILabel] = classCount.get(voteILabel,0) + 1
        print(voteILabel,"aa", classCount[voteILabel])
    maxLabel = sorted(classCount.items(), key = operator.itemgetter(1),reverse = True)
    print(operator.itemgetter(1),classCount.items(),maxLabel)
    return maxLabel[0][0]

if __name__=="__main__":
    g,l = creatDataSet()
    labelOfinX = classify([1.0,1.2],g,l,1)
    print(labelOfinX)