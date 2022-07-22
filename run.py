import glob
import os
import re
import sys
import argparse
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from LeNet5 import LeNet5
from AlexNet8 import AlexNet8
from GoogLeNet import GoogLeNet
from VGG import VGG16, VGG19
from ResNet import Shallow_Res, Deep_Res
from datasets import load_data


class_number = 10  # 几分类任务
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


"""
    本函数代码来自YOLOv5    源码Github地址:https://github.com/ultralytics/yolov5
"""
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


'''
    运行函数
    参数：model_name:字符串，模型算法名，可选Lenet5、Alexnet8、VGG16、VGG19、Res18、Res34、Res50、Res101、Res152
    train_data：元组或列表，（训练集，训练集标签）形式
    validation_data：元组或列表，（测试集，测试集标签）形式
    batch_size：int，批次大小，默认为128
    epochs：int，迭代次数，默认为100

    函数运行首先会检测模型是否已存在，存在则直接加载模型模型进行训练；不存在则创建模型。训练完成后会输出模型结构
    可视化训练和测试的acc和loss曲线，并保存最优模型和模型的权重参数。
'''
def run(model='Res34',  # 要运行的模型  可选Lenet5, Alexnet8, GoogLeNet, VGG16, VGG19, Res18, Res34, Res50, Res101, Res152
        dataset='images_set',  # 使用那个数据集  可选 images_set, cifar10
        optimizer='adam',  # 使用的优化器  如sgd, adam, adagrad, nadam, rmsprop
        batch_size=128,  # 批次大小
        epochs=100,  # 迭代次数
        project=ROOT / 'runs',  # checkpoint文件和weights文件保存目录
        name='weights',  # weights文件目录
        exist_ok=False  # existing project/name ok, do not increment
        ):
    checkpoint_save_path = project / "checkpoint/{}.ckpt".format(model)
    save_dir = increment_path(project / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)   # create dir
    weights_save_path = save_dir / '{}_weights.txt'.format(model)
    models = {
        "Lenet5": LeNet5(class_number),
        "Alexnet8": AlexNet8(class_number),
        "GoogLeNet":GoogLeNet(class_number),
        "VGG16": VGG16(class_number),
        "VGG19": VGG19(class_number),
        "Res18": Shallow_Res([2, 2, 2, 2], class_number),
        "Res34": Shallow_Res([3, 4, 6, 3], class_number),
        "Res50": Deep_Res([3, 4, 6, 3], class_number),
        "Res101": Deep_Res([3, 4, 23, 3], class_number),
        "Res152": Deep_Res([3, 8, 36, 3], class_number)
    }
    x_train, y_train, x_test, y_test = load_data(data_set=dataset)
    model = models[model]
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    if os.path.exists('{}.index'.format(checkpoint_save_path)):
        print('---' * 20, ' load the model... ', '---' * 20)
        model.load_weights(checkpoint_save_path)
    op_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, save_weights_only=True,
                                                     save_best_only=True)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(x_test, y_test), validation_freq=1, callbacks=[op_callback])
    model.summary()

    # 可视化 loss 和 acc
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title("Training and Valiidation Loss")
    plt.legend()
    plt.show()

    # 模型参数写入文件 ./weights.txt
    with open(weights_save_path, 'w') as file:
        for v in model.trainable_variables:
            file.write(str(v.name) + '\n')
            file.write(str(v.shape) + '\n')
            file.write(str(v.numpy()) + '\n')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Res34', help='run model')
    parser.add_argument('--dataset', type=str, default='images_set', help='use dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='size of a batch')
    parser.add_argument('--epochs', type=int, default=100, help='number of one epoch')
    parser.add_argument('--project', default=ROOT / 'runs', help='save results to project path')
    parser.add_argument('--name', default='weights', help='save model weights to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))
