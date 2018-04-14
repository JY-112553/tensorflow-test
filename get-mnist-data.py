import numpy as np
import input_data
import cv2


def _load_data():
    # 加载数据
    data = input_data.read_data_sets('MNIST_data/', one_hot=True)
    return data


def byte2img(data, path):
    trX, teX = data.train.images, data.test.images
    # 把二进制格式的图像images保存为jpg格式
    for i in range(len(trX)):
        img = 255 * np.mat(trX[i]).reshape(28, 28)
        cv2.imwrite(path + 'train/img/' + str(i + 1) + '.jpg', img)
    for i in range(len(teX)):
        img = 255 * np.mat(teX[i]).reshape(28, 28)
        cv2.imwrite(path + 'test/img/' + str(i + 1) + '.jpg', img)


def byte2txt(data, path):
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
    path = '/media/sf_Datasets/MNIST/'  # 文集保存路径
    byte2img(mnist_data, path)
    byte2txt(mnist_data, path)


if __name__ == '__main__':
    main()
