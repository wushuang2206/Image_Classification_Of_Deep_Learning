import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10


# 每个标签对应类名
features = []
'''
    读取数据
    参数classes：int, 类别数量，最多为17类
        train_size：int, 训练及图片数，当图片不足时，为读取本类别所有图片
        test_szie：int, 测试集样本数量
        优先将样本划分给测试集，当测试集够了后，再将样本添加到训练集。如有1500张图，先划分给测试集128张，剩余划分给训练集
        三个参数的默认值为17，1500，128。当所有都为默认值即读取完所有images_set的图片
    函数返回：x_train, y_train, x_test, y_test
            训练集、训练集标签、测试集、测试集标签（标签为数值，如0，1，2……）
'''
def load_images_set(classes=17, train_size=1500, test_size=128):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    img_label = 0
    for img_dir in os.listdir("./images_set/"):
        features.append(img_dir)
        num = 0
        for img_path in os.listdir("./images_set/" + str(img_dir)):
            img = cv2.imread("./images_set/" + str(img_dir) + '/' + str(img_path))
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            '''
                转换为 224*224 (高，宽) 如(400, 500) 输出(500, 400)
                原算法用的是 224*224，内存足够的推荐使用224*224
                个人配置内存不足，所以转为更小的size  112*112
            '''
            # img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
            # print(img.shape)
            if num < test_size:
                x_test.append(img)
                y_test.append(img_label)
            elif num < (train_size + test_size):
                x_train.append(img)
                y_train.append(img_label)
            else:
                break
            num += 1
        img_label += 1
        if img_label == classes:
            break

    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    return x_train, y_train, x_test, y_test


'''
    加载数据集
    参数：data_set:字符串，可选"cifar10"、"images_set"
    函数返回：x_train, y_train, x_test, y_test
            训练集、训练集标签、测试集、测试集标签（标签为数值，如0，1，2……）
    功能：加载images_set数据集或cifar10数据集，并完成数据归一化和数据乱序操作1
'''
def load_data(data_set, class_number=17):
    if data_set == 'images_set':
        print('---' * 20, ' loading images_set... ', '---' * 20)
        x_train, y_train, x_test, y_test = load_images_set(classes=class_number, train_size=1500)

        # 查看前10张图片
        fig = plt.figure(figsize=(25, 25))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(x_train[i])
            plt.title(features[int(y_train[i])])
            fig.tight_layout(pad=3.0)
        plt.show()

        # 检查数据集总体类分布
        df = pd.DataFrame(data=np.hstack((y_train, y_test)))
        # print(df)
        counts = df.value_counts().sort_index()
        print(counts)

        def class_distribution(x, y, labels):
            fig, ax = plt.subplots()
            ax.bar(x, y)
            ax.set_xticklabels(labels, rotation=90)
            plt.show()

        class_distribution(features, counts, features)
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    x_train, x_test = x_train / 255.0, x_test / 255.0
    # 训练集乱序
    np.random.seed(116)
    np.random.shuffle(x_train)
    np.random.seed(116)
    np.random.shuffle(y_train)
    tf.random.set_seed(116)
    # 测试集乱序
    np.random.seed(66)
    np.random.shuffle(x_test)
    np.random.seed(66)
    np.random.shuffle(y_test)
    print('***'*20, '数据归一化、数据集乱序已完成...', '***'*20)
    return x_train, y_train, x_test, y_test
