import tensorflow as tf
import numpy as np

img_size_x = 224
img_size_y = 224
img_chanel = 3


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
    image = tf.reshape(image, [img_size_x, img_size_y, img_chanel])
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.string)

    return image, label


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w1a, w1b, w2a, w2b, w3a, w3b, w3c,
          w4a, w4b, w4c, w5a, w5b, w5c, w6, w7, w_o):
    # 第一组卷积层及池化层
    l1a = tf.nn.relu(tf.nn.conv2d(X, w1a, strides=[1, 1, 1, 1], padding='SAME'))
    # l1a shape=(?, 224, 224, 64)
    l1b = tf.nn.relu(tf.nn.conv2d(l1a, w1b, strides=[1, 1, 1, 1], padding='SAME'))
    # l1b shape=(?, 224, 224, 64)
    l1 = tf.nn.max_pool(l1b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # l1 shape=(?, 112, 112, 64)

    # 第二组卷积层及池化层
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2a, strides=[1, 1, 1, 1], padding='SAME'))
    # l2a shape=(?, 112, 112, 128)
    l2b = tf.nn.relu(tf.nn.conv2d(l2a, w2b, strides=[1, 1, 1, 1], padding='SAME'))
    # l2b shape=(?, 112, 112, 128)
    l2 = tf.nn.max_pool(l2b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # l2 shape=(?, 56, 56, 128)

    # 第三组卷积层及池化层
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3a, strides=[1, 1, 1, 1], padding='SAME'))
    # l3a shape=(?, 56, 56, 256)
    l3b = tf.nn.relu(tf.nn.conv2d(l3a, w3b, strides=[1, 1, 1, 1], padding='SAME'))
    # l3b shape=(?, 56, 56, 256)
    l3c = tf.nn.relu(tf.nn.conv2d(l3b, w3c, strides=[1, 1, 1, 1], padding='SAME'))
    # l3b shape=(?, 56, 56, 256)
    l3 = tf.nn.max_pool(l3c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # l3 shape=(?, 28, 28, 256)

    # 第四组卷积层及池化层
    l4a = tf.nn.relu(tf.nn.conv2d(l3, w4a, strides=[1, 1, 1, 1], padding='SAME'))
    # l4a shape=(?, 28, 28, 512)
    l4b = tf.nn.relu(tf.nn.conv2d(l4a, w4b, strides=[1, 1, 1, 1], padding='SAME'))
    # l4b shape=(?, 28, 28, 512)
    l4c = tf.nn.relu(tf.nn.conv2d(l4b, w4c, strides=[1, 1, 1, 1], padding='SAME'))
    # l4c shape=(?, 28, 28, 512)
    l4 = tf.nn.max_pool(l4c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # l4 shape=(?, 14, 14, 512)

    # 第五组卷积层及池化层
    l5a = tf.nn.relu(tf.nn.conv2d(l4, w5a, strides=[1, 1, 1, 1], padding='SAME'))
    # l5a shape=(?, 14, 14, 512)
    l5b = tf.nn.relu(tf.nn.conv2d(l5a, w5b, strides=[1, 1, 1, 1], padding='SAME'))
    # l5b shape=(?, 14, 14, 512)
    l5c = tf.nn.relu(tf.nn.conv2d(l5b, w5c, strides=[1, 1, 1, 1], padding='SAME'))
    # l5c shape=(?, 14, 14, 512)
    l5 = tf.nn.max_pool(l5c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # l5 shape=(?, 7, 7, 512)
    l5 = tf.reshape(l5, [-1, w6.get_shape().as_list()[0]])
    # reshape to (?, 512 * 7 * 7)

    # 全连接层
    l6 = tf.nn.relu(tf.matmul(l5, w6))
    # l6 shape=(?, 4096)

    # 全连接层
    l7 = tf.nn.relu(tf.matmul(l6, w7))
    # l7 shape=(?, 4096)

    # 输出层
    py = tf.matmul(l7, w_o)
    # py shape=(?, 10)

    return py  # 返回预测值


def _train(path_train, filename_train, path_test, filename_test):
    images_train, labels_train = readTFRecord(path_train + filename_train + '.tfrecords')
    images_test, labels_test = readTFRecord(path_test + filename_test + '.tfrecords')
    train_size = 8
    test_size = 8

    images_train_batch, labels_train_batch = \
        tf.train.shuffle_batch([images_train, labels_train],
                               batch_size=train_size, capacity=2000, min_after_dequeue=1000)
    images_test_batch, labels_test_batch = \
        tf.train.shuffle_batch([images_test, labels_test],
                               batch_size=test_size, capacity=2000, min_after_dequeue=1000)

    X = tf.placeholder("float", [None, img_size_x, img_size_y, img_chanel])
    Y = tf.placeholder("float", [None, 10])

    w1a = init_weights([3, 3, 3, 64])  # patch 大小为 3 × 3 ,输入维度为 3 ,输出维度为 64
    w1b = init_weights([3, 3, 64, 64])  # patch 大小为 3 × 3 ,输入维度为 64 ,输出维度为 64
    w2a = init_weights([3, 3, 64, 128])  # patch 大小为 3 × 3 ,输入维度为 64 ,输出维度为 128
    w2b = init_weights([3, 3, 128, 128])  # patch 大小为 3 × 3 ,输入维度为 128 ,输出维度为 128
    w3a = init_weights([3, 3, 128, 256])  # patch 大小为 3 × 3 ,输入维度为 128 ,输出维度为 256
    w3b = init_weights([3, 3, 256, 256])  # patch 大小为 3 × 3 ,输入维度为 256 ,输出维度为 256
    w3c = init_weights([3, 3, 256, 256])  # patch 大小为 3 × 3 ,输入维度为 256 ,输出维度为 256
    w4a = init_weights([3, 3, 256, 512])  # patch 大小为 3 × 3 ,输入维度为 256 ,输出维度为 512
    w4b = init_weights([3, 3, 512, 512])  # patch 大小为 3 × 3 ,输入维度为 256 ,输出维度为 256
    w4c = init_weights([3, 3, 512, 512])  # patch 大小为 3 × 3 ,输入维度为 256 ,输出维度为 256
    w5a = init_weights([3, 3, 512, 512])  # patch 大小为 3 × 3 ,输入维度为 256 ,输出维度为 256
    w5b = init_weights([3, 3, 512, 512])  # patch 大小为 3 × 3 ,输入维度为 256 ,输出维度为 256
    w5c = init_weights([3, 3, 512, 512])  # patch 大小为 3 × 3 ,输入维度为 256 ,输出维度为 256
    w6 = init_weights([512 * 7 * 7, 4096])  # 全连接层,输入维度为 512 × 7 × 7 ,输出维度为4096
    w7 = init_weights(([4096, 4096]))  # 全连接层,输入维度为 4096, 输出维度为 4096
    w_o = init_weights([4096, 10])  # 输出层,输入维度为 4096, 输出维度为 10 ,代表 10 类 (labels)

    py_x = model(X, w1a, w1b, w2a, w2b, w3a, w3b, w3c,
                 w4a, w4b, w4c, w5a, w5b, w5c, w6, w7, w_o)  # 得到预测值

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=py_x, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        train_step = 0
        train_steps = 2000

        try:
            while not coord.should_stop():  # 如果线程应该停止则返回True
                if train_step >= train_steps:
                    coord.request_stop()  # 请求该线程停止

                trX, trY, teX, teY = sess.run([images_train_batch, labels_train_batch,
                                               images_test_batch, labels_test_batch])
                trX = trX.reshape(-1, img_size_x, img_size_y, img_chanel)  # 224x224x3 input img
                teX = teX.reshape(-1, img_size_x, img_size_y, img_chanel)  # 224x224x3 input img

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

    path_size = '(' + str(img_size_x) + 'x' + str(img_size_y) + ')'

    filename_train = 'MNIST_train' + path_size
    filename_test = 'MNIST_test' + path_size
    _train(path_train, filename_train, path_test, filename_test)


if __name__ == '__main__':
    main()
