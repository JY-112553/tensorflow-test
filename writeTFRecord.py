import tensorflow as tf
import cv2
import os


def writeTFRecord(path, filename):
    images_list = []
    labels_list = []

    # 生成一个txt文件来记录images路径和labels，生成txt仅为了方便调试代码，可以省去生成txt直接创建list
    with open(path + filename + '.txt', 'w') as f:  # txt文件生成地址
        train_file_len = len(os.listdir(path + 'img'))
        for i in range(1, train_file_len + 1):
            with open(path + 'label/' + str(i) + '.txt', 'r') as fl:
                label = fl.read()
            # images路径和labels用，做区分
            f.write(path + 'img/' + str(i) + '.jpg' + ',' + label + '\n')

    # 生成list，用于生成tfrecords文件
    with open(path + filename + '.txt', 'r') as f:
        for line in f.readlines():
            images_list.append(line.split(',')[0])  # images路径列表
            labels_list.append(line.split(',')[1])  # labels列表

    # 生成tfrecords文件
    writer = tf.python_io.TFRecordWriter(path + filename + '.tfrecords')
    for i, [image, label] in enumerate(zip(images_list, labels_list)):
        _image = cv2.imread(image)  # 使用opencv读取图像数据
        _image = cv2.cvtColor(_image, cv2.COLOR_RGB2GRAY)  # 因为图像保存时为三通到图像，此处要转化为灰度图像
        image_raw = _image.tobytes()  # 图像矩阵转化为bytes
        label = str.encode(label)  # 字符串转化为bytes

        # 写入tfrecords
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
        }))
        writer.write(example.SerializeToString())
    writer.close()


def main():
    path = os.getcwd() + '/MNIST_data/'  # 文件保存路径
    path_train = path + 'train/'
    path_test = path + 'test/'
    filename_train = 'MNIST_train'
    filename_test = 'MNIST_test'
    writeTFRecord(path_train, filename_train)
    writeTFRecord(path_test, filename_test)


if __name__ == '__main__':
    main()
