import math

def F(x):
    return (math.exp(abs(x)) - 1) / (math.exp(abs(x)) + 1)


X = list(range(1, 11))
Y = list(range(10, 110, 10))

k, b = 0, 0
for repeat in range(1000000):
    for i in range(10):
        c1 = k * X[i]
        e1 = Y[i] - b - c1
        d1 = e1 * F(e1)
        k += d1
        c2 = k * X[i] + b
        e2 = Y[i] - c2
        d2 = e2 * F(e2)
        b += d2

print(k, b)