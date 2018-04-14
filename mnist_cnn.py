import tensorflow as tf
import numpy as np
import input_data


def _load_data():
    # 加载数据
    data = input_data.read_data_sets('MNIST_data/', one_hot=True)
    return data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    # 第一组卷积层及池化层,最后 dropout 一些神经元
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    # l1a shape=(?, 28, 28, 32)
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # l1 shape=(?, 14, 14, 32)
    l1 = tf.nn.dropout(l1, p_keep_conv)

    # 第二组卷积层及池化层,最后 dropout 一些神经元
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    # l2a shape=(?, 14, 14, 64)
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # l2 shape=(?, 7, 7, 64)
    l2 = tf.nn.dropout(l2, p_keep_conv)

    # 第三组卷积层及池化层,最后 dropout 一些神经元
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
    # l3a shape=(?, 7, 7, 128)
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # l3 shape=(?, 4, 4, 128)
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    # 全连接层,最后 dropout 一些神经元
    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

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

    w = init_weights([3, 3, 1, 32])  # patch 大小为 3 × 3 ,输入维度为 1 ,输出维度为 32
    w2 = init_weights([3, 3, 32, 64])  # patch 大小为 3 × 3 ,输入维度为 32 ,输出维度为 64
    w3 = init_weights([3, 3, 64, 128])  # patch 大小为 3 × 3 ,输入维度为 64 ,输出维度为 128
    w4 = init_weights([128 * 4 * 4, 625])  # 全连接层,输入维度为 128 × 4 × 4,
    # 是上一层的输出数据又三维的转变成一维, 输出维度为625
    w_o = init_weights([625, 10])  # 输出层,输入维度为 625, 输出维度为 10 ,代表 10 类 (labels)

    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
    py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)  # 得到预测值

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
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                              p_keep_conv: 0.8, p_keep_hidden: 0.5})
            test_indices = np.arange(len(teX))  # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]
            predict = sess.run(predict_op, feed_dict={X: teX[test_indices], p_keep_conv: 1.0, p_keep_hidden: 1.0})
            print(i, np.mean(np.argmax(teY[test_indices], axis=1) == predict))


def main():
    mnist_data = _load_data()
    _train(mnist_data)


if __name__ == '__main__':
    main()
