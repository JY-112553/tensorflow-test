# tensorflow-test
使用指南：
1.使用get_mnist_data.py获取tensorflow官方的mnist数据，解析并保存为jpg格式的图片和txt格式的标签。针对不同模型，图片的尺寸需要改变。
2.使用writetfrecords.py把图片和标签写入tfrecords格式文件，使用tfrecords是一种比较通用、高效的数据读取方法，是tensorflow官方推荐的标准格式。tfrecords文件是一种将图像数据和标签统一存储的二进制文件，能更好的利用内存，在tensorflow中快速的复制，移动，读取，存储等。
3.使用各类模型读取相应的tfrecords文件，查看训练效果。
 
18-04-23
基于tensorflow官方的mnist数据，实现了图片标签数据向tfrecords的转换。对tfrecords格式的mnist数据使用LeNet模型。

18-04-27
修改部分代码。

18-04-28
修改部分代码。对tfrecords格式的mnist数据使用AlexNet模型。
