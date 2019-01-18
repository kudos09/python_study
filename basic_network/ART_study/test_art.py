# -*- coding: utf-8 -*-
# 这段代码主要的功能是把utf-8格式的json文件转换成excel表格

def ART1(self, trainingPattern, isTraining):
    inputSum = 25

    activationSum = 4

    f2Max = 0.7

    reset = True

    for i in range(self.mNumClusters):
        self.f2[i] = 0.0

    for i in range(self.mInputSize):
        self.f1a[i] = float(trainingPattern[i])

    inputSum = self.get_vector_sum(self.f1a)

    for i in range(self.mInputSize):
        self.f1b[i] = self.f1a[i]

    for i in range(self.mNumClusters):

        for j in range(self.mInputSize):
            self.f2[i] += self.bw[i][j] * float(self.f1a[j])

    reset = True

    while reset:

        f2Max = self.get_maximum(self.f2)

        if f2Max == -1:
            f2Max = self.mNumClusters

            self.f2.append(0.0)

            self.tw.append([1.0] * self.mInputSize)

            self.bw.append([1.0 / (1.0 + self.mInputSize)] * self.mInputSize)

            self.mNumClusters += 1

        for i in range(self.mInputSize):
            self.f1b[i] = self.f1a[i] * math.floor(self.tw[f2Max][i])

        activationSum = self.get_vector_sum(self.f1b)

        reset = self.test_for_reset(activationSum, inputSum, f2Max)

    if isTraining:
        self.update_weights(activationSum, f2Max)

    return f2Max



if __name__ == '__main__':

    VIGILANCE = 0.3
    PATTERN_ARRAY = [[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
                     [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
                     [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]]
    ART1(self, PATTERN_ARRAY, VIGILANCE)
print ("json to excel OK")