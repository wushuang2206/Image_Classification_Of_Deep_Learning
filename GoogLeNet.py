from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, Concatenate, BatchNormalization, GlobalAveragePooling2D, Dropout, Dense
'''
    Concatenate用法:
        >>> x = np.arange(20).reshape(2, 2, 5)
        >>> print(x)
        [[[ 0  1  2  3  4]
          [ 5  6  7  8  9]]
         [[10 11 12 13 14]
          [15 16 17 18 19]]]
        >>> y = np.arange(20, 30).reshape(2, 1, 5)
        >>> print(y)
        [[[20 21 22 23 24]]
         [[25 26 27 28 29]]]
        >>> tf.keras.layers.Concatenate(axis=1)([x, y])
        <tf.Tensor: shape=(2, 3, 5), dtype=int64, numpy=
        array([[[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [20, 21, 22, 23, 24]],
               [[10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [25, 26, 27, 28, 29]]])>
        形参:
        axis – Axis along which to concatenate.  axis为0表示第1维，为1表示第2维，依次类推。默认为-1，最后一维
        kwargs – standard layer keyword arguments.

'''


class Inception(Model):
    def __init__(self, filters_list):
        super(Inception, self).__init__()
        # 1*1卷积部分
        self.conv1 = Conv2D(filters=filters_list[0], kernel_size=(1, 1), strides=1)
        self.BN1 = BatchNormalization()
        self.relu1 = Activation('relu')
        # 1*1 -->> 3*3 卷积部分
        self.conv2_1 = Conv2D(filters=filters_list[1], kernel_size=(1, 1), strides=1)
        self.BN2_1 = BatchNormalization()
        self.relu2_1 = Activation('relu')
        self.conv2_2 = Conv2D(filters=filters_list[2], kernel_size=(3, 3), strides=1, padding='same')
        self.BN2_2 = BatchNormalization()
        self.relu2_2 = Activation('relu')
        # 1*1 -->> 5*5 卷积部分
        self.conv3_1 = Conv2D(filters=filters_list[3], kernel_size=(1, 1), strides=1)
        self.BN3_1 = BatchNormalization()
        self.relu3_1 = Activation('relu')
        self.conv3_2 = Conv2D(filters=filters_list[4], kernel_size=(5, 5), strides=1, padding='same')
        self.BN3_2 = BatchNormalization()
        self.relu3_2 = Activation('relu')
        # MaxPool -->> 1*1 部分
        self.pool = MaxPool2D((3, 3), strides=1, padding='same')
        self.conv4_1 = Conv2D(filters=filters_list[5], kernel_size=(1, 1), strides=1)
        self.BN4_1 = BatchNormalization()
        self.relu4_1 = Activation('relu')

    def call(self, inputs):
        x = inputs
        # 第1部分卷积操作 1*1
        out1 = self.conv1(x)
        out1 = self.BN1(out1)
        out1 = self.relu1(out1)
        # 第2部分卷积操作 3*3
        out2 = self.conv2_1(x)
        out2 = self.BN2_1(out2)
        out2 = self.relu2_1(out2)
        out2 = self.conv2_2(out2)
        out2 = self.BN2_2(out2)
        out2 = self.relu2_2(out2)
        # 第3部分卷积操作 5*5
        out3 = self.conv3_1(x)
        out3 = self.BN3_1(out3)
        out3 = self.relu3_1(out3)
        out3 = self.conv3_2(out3)
        out3 = self.BN3_2(out3)
        out3 = self.relu3_2(out3)
        # 第4部分 池化卷积操作
        out4 = self.pool(x)
        out4 = self.conv4_1(out4)
        out4 = self.BN4_1(out4)
        out4 = self.relu4_1(out4)
        # DepthConcat
        output = Concatenate(axis=-1)([out1, out2, out3, out4])
        print(output.shape)

        return output


'''
    实现中去除了原论文中的两个辅助训练器（inception_4a和inception_4d后面的softmax分支）
'''
class GoogLeNet(Model):
    def __init__(self, class_number):
        super(GoogLeNet, self).__init__()
        self.conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=2)
        self.pool1 = MaxPool2D((3, 3), strides=2)
        self.BN1 = BatchNormalization()
        self.relu1 = Activation('relu')
        self.conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=1)
        self.conv3 = Conv2D(filters=192, kernel_size=(3, 3), strides=1)
        self.BN2 = BatchNormalization()
        self.relu2 = Activation('relu')
        self.pool2 = MaxPool2D((3, 3), strides=2)
        self.inception_3a = Inception([64, 96, 128, 16, 32, 32])
        self.inception_3b = Inception([128, 128, 192, 32, 96, 64])
        self.pool3 = MaxPool2D((3, 3), strides=2)
        self.inception_4a = Inception([192, 96, 208, 16, 48, 64])
        self.inception_4b = Inception([160, 112, 224, 24, 64, 64])
        self.inception_4c = Inception([128, 128, 256, 24, 64, 64])
        self.inception_4d = Inception([112, 144, 288, 32, 64, 64])
        self.inception_4e = Inception([256, 160, 320, 32, 128, 128])
        self.pool4 = MaxPool2D((3, 3), strides=2)
        self.inception_5a = Inception([256, 160, 320, 32, 128, 128])
        self.inception_5b = Inception([384, 192, 384, 48, 128, 128])
        self.avg_pool = GlobalAveragePooling2D()
        self.dropout = Dropout(0.4)
        self.linear = Dense(1000, activation='relu')
        self.FC = Dense(class_number, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.BN1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.BN2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.pool3(x)
        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        x = self.pool4(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.linear(x)
        y = self.FC(x)

        return y

