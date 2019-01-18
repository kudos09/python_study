import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(1, 100))  # 随机输入
y_data = np.dot([10], x_data) + 0

# 构造一个线性模型
b = tf.Variable(tf.zeros([1])) + 0.1
W = tf.Variable(tf.random_uniform([1, 1], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, 301):
    sess.run(train)
    if step % 30 == 0:
        print(step, sess.run(W), sess.run(b), sess.run(loss))