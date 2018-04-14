import tensorflow as tf
import numpy as np
import input_data


def _load_data():
    # 加载数据
    data = input_data.read_data_sets('MNIST_data/', one_hot=True)
    return data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o):
    # 第一组卷积层及池化层
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    # l1a shape=(?, 28, 28, 6)
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # l1 shape=(?, 14, 14, 6)

    # 第二组卷积层及池化层
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='VALID'))
    # l2a shape=(?, 10, 10, 16)
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # l2 shape=(?, 5, 5, 16)
    l2 = tf.reshape(l2, [-1, w3.get_shape().as_list()[0]])
    # reshape to (?, 16 * 5 * 5)

    # 全连接层
    l3 = tf.nn.relu(tf.matmul(l2, w3))  # l3 shape=(?, 120)

    # 全连接层
    l4 = tf.nn.relu(tf.matmul(l3, w4))  # l4 shape=(?, 84)

    # 输出层
    pyx = tf.matmul(l4, w_o)

    return pyx  # 返回预测值


def _train(data):
    trX, trY, teX, teY = data.train.images, data.train.labels, \
        data.test.images, data.test.labels
    trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
    teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

    X = tf.placeholder("float", [None, 28, 28, 1])
    Y = tf.placeholder("float", [None, 10])

    w = init_weights([5, 5, 1, 6])  # patch 大小为 5 × 5 ,输入维度为 1 ,输出维度为 6
    w2 = init_weights([5, 5, 6, 16])  # patch 大小为 5 × 5 ,输入维度为 6 ,输出维度为 16
    w3 = init_weights([16 * 5 * 5, 120])  # 全连接层,输入维度为 16 × 5 × 5,
    # 是上一层的输出数据又三维的转变成一维, 输出维度为120
    w4 = init_weights(([120, 84]))  # 全连接层,输入维度为 625, 输出维度为 10
    w_o = init_weights([84, 10])  # 输出层,输入维度为 625, 输出维度为 10 ,代表 10 类 (labels)

    py_x = model(X, w, w2, w3, w4, w_o)  # 得到预测值

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=py_x, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    batch_size = 128
    test_size = 256

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        for i in range(100):
            training_batch = zip(range(0, len(trX), batch_size),
                                 range(batch_size, len(trX) + 1, batch_size))
            for start, end in training_batch:
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
            test_indices = np.arange(len(teX))  # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]
            predict = sess.run(predict_op, feed_dict={X: teX[test_indices]})
            print(i, np.mean(np.argmax(teY[test_indices], axis=1) == predict))


def main():
    mnist_data = _load_data()
    _train(mnist_data)


if __name__ == '__main__':
    main()
