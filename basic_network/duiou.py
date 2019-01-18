
import sys

class dualPerceptron(object):

    def __init__(self):
        self.learning_rate = 1
        self.epoch = 10

    def train(self, features, labels):
        self.alpha = [0.0] * (len(features))
        self.bias = 0.0

        self.gram = [[0 for i in range(len(features))] for j in range(len(features))]

        print('calc gram')
        # calc gram matrix
        for i in range(len(features)):
            for j in range(len(features)):
                sum = 0.0
                for k in range(len(features[0])):
                    sum += features[i][k] * features[j][k]
                self.gram[i][j] = sum
        print('gram over')

        print(self.gram)

        idx = 0

        while idx < self.epoch:
            idx += 1
            print('epoch: {}'.format(idx))
            print( self.alpha)
            print( self.bias)
            for i in range(len(features)):
                yi = labels[i]

                sum = 0.0
                for j in range(len(features)):
                    yj = labels[j]
                    sum += self.alpha[j] * yj * self.gram[j][i]

                if yi * (sum + self.bias) <= 0:
                    self.alpha[i] = self.alpha[i] + self.learning_rate
                    self.bias = self.bias + self.learning_rate * yi

        print( self.alpha)
        print( self.bias)


if __name__ == '__main__':
    p = dualPerceptron()
    data = [[3, 3, ], [4, 3], [1, 1]]
    label = [1, 1, -1]
    p.train(data, label)
