import tensorflow as tf
import numpy as np


def readTFRecord(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.string)
                                       })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [28, 28])
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.string)

    return image, label


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


def _train(path_train, filename_train, path_test, filename_test):
    images_train, labels_train = readTFRecord(path_train + filename_train + '.tfrecords')
    images_test, labels_test = readTFRecord(path_test + filename_test + '.tfrecords')
    train_size = 128
    test_size = 256

    images_train_batch, labels_train_batch = \
        tf.train.shuffle_batch([images_train, labels_train],
                               batch_size=train_size, capacity=2000, min_after_dequeue=1000)
    images_test_batch, labels_test_batch = \
        tf.train.shuffle_batch([images_test, labels_test],
                               batch_size=test_size, capacity=2000, min_after_dequeue=1000)

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

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        train_step = 0
        train_steps = 2000
        tf.global_variables_initializer().run()

        try:
            while not coord.should_stop():  # 如果线程应该停止则返回True
                if train_step >= train_steps:
                    coord.request_stop()  # 请求该线程停止

                trX, trY, teX, teY = sess.run([images_train_batch, labels_train_batch,
                                               images_test_batch, labels_test_batch])
                trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
                teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

                temp1 = []
                temp2 = []
                for i in range(trY.shape[0]):
                    _trY = trY[i]
                    _trY = _trY.decode().split(' ')
                    _trY.pop()
                    temp1.append(_trY)
                _trY = np.array(temp1)
                for i in range(teY.shape[0]):
                    _teY = teY[i]
                    _teY = _teY.decode().split(' ')
                    _teY.pop()
                    temp2.append(_teY)
                _teY = np.array(temp2)

                sess.run(train_op, feed_dict={X: trX, Y: _trY})
                predict = sess.run(predict_op, feed_dict={X: teX})
                if train_step % 100 == 0:
                    print(train_step, np.mean(np.argmax(_teY, axis=1) == predict))
                train_step += 1

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop. 请求该线程停止
            coord.request_stop()
            # And wait for them to actually do it. 等待被指定的线程终止
            coord.join(threads)


def main():
    path = 'D:/Datasets/MNIST/'   # 文件保存路径
    path_train = path + 'train/'
    path_test = path + 'test/'
    filename_train = 'MNIST_train'
    filename_test = 'MNIST_test'
    _train(path_train, filename_train, path_test, filename_test)


if __name__ == '__main__':
    main()
