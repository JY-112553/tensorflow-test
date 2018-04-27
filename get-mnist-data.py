import numpy as np
import input_data
import cv2
import os


def _load_data():
    # 加载数据
    data = input_data.read_data_sets('MNIST_data/', one_hot=True)
    return data


def mkdir(path):
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        print(path + '创建成功')
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + '目录已存在')
        return False


def byte2img(data, path):
    mkdir(path + 'train/img')
    mkdir(path + 'test/img')

    trX, teX = data.train.images, data.test.images
    # 把二进制格式的图像images保存为jpg格式
    for i in range(len(trX)):
        img = 255 * np.mat(trX[i]).reshape(28, 28)
        # img = cv2.resize(img, (32, 32))  # 改变图像尺寸
        cv2.imwrite(path + 'train/img/' + str(i + 1) + '.jpg', img)
    for i in range(len(teX)):
        img = 255 * np.mat(teX[i]).reshape(28, 28)
        # img = cv2.resize(img, (32, 32))  # 改变图像尺寸
        cv2.imwrite(path + 'test/img/' + str(i + 1) + '.jpg', img)


def byte2txt(data, path):
    mkdir(path + 'train/label')
    mkdir(path + 'test/label')

    trY, teY = data.train.labels, data.test.labels
    # 把二进制格式的标签labels保存为txt格式
    for i in range(len(trY)):
        np.savetxt(path + 'train/label/' + str(i + 1) + '.txt',
                   trY[i], fmt='%0.1f', newline=' ')
    for i in range(len(teY)):
        np.savetxt(path + 'test/label/' + str(i + 1) + '.txt',
                   teY[i], fmt='%0.1f', newline=' ')


def main():
    mnist_data = _load_data()
    path = os.getcwd() + '/MNIST_data/'  # 文件保存路径
    byte2img(mnist_data, path)
    byte2txt(mnist_data, path)


if __name__ == '__main__':
    main()
