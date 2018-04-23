import tensorflow as tf
import cv2
import os


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


def method(path, filename):
    # tfrecords的使用方法
    images, labels = readTFRecord(path + filename + '.tfrecords')
    images_batch, labels_batch = tf.train.shuffle_batch([images, labels],
                                                        batch_size=10, capacity=2000, min_after_dequeue=1000)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        train_steps = 10

        try:
            while not coord.should_stop():  # 如果线程应该停止则返回True
                images, labels = sess.run([images_batch, labels_batch])

                for i in range(images.shape[0]):
                    image = images[i]
                    cv2.imshow('img', image)
                    cv2.waitKey(0)
                print(images.shape, labels)

                train_steps -= 1
                print(train_steps)
                if train_steps <= 0:
                    coord.request_stop()  # 请求该线程停止

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop. 请求该线程停止
            coord.request_stop()
            # And wait for them to actually do it. 等待被指定的线程终止
            coord.join(threads)


def main():
    path = os.getcwd() + '/MNIST_data/'  # 文件读取路径
    path_train = path + 'train/'
    path_test = path + 'test/'
    filename_train = 'MNIST_train'
    filename_test = 'MNIST_test'

    method(path_train, filename_train)


if __name__ == '__main__':
    main()
