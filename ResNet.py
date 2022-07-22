from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Dense


# res18、res34 class
class Shallow_ResBlock(Model):
    def __init__(self, filters, strides=1, residual_dimension=False):
        super(Shallow_ResBlock, self).__init__()
        self.residual_dimension = residual_dimension

        self.conv1 = Conv2D(filters, (3, 3), strides=strides, padding='same')
        self.conv2 = Conv2D(filters, (3, 3), strides=1, padding='same')
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.relu1 = Activation('relu')
        self.relu2 = Activation('relu')

        if residual_dimension:
            self.residual_conv = Conv2D(filters, (1, 1), strides=strides)
            self.residual_bn = BatchNormalization()

    def call(self, inputs):
        residual_X = inputs

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        Fx = self.bn2(x)

        if self.residual_dimension:
            residual_X = self.residual_conv(residual_X)
            residual_X = self.residual_bn(residual_X)
        y = self.relu2(Fx + residual_X)

        return y


class Shallow_Res(Model):
    def __init__(self, blocks_list, class_number):
        super(Shallow_Res, self).__init__()
        self.blocks_num = len(blocks_list)
        self.blocks_list = blocks_list

        self.conv = Conv2D(64, (7, 7), strides=2)
        self.bn = BatchNormalization()
        self.relu = Activation('relu')
        self.pool = MaxPool2D((3, 3), strides=2)

        self.blocks = Sequential()
        self.filters = 64
        for block_id in range(self.blocks_num):
            for layer_id in range(blocks_list[block_id]):
                if block_id != 0 and layer_id == 0:
                    block = Shallow_ResBlock(self.filters, strides=2, residual_dimension=True)
                else:
                    block = Shallow_ResBlock(self.filters)
                self.blocks.add(block)
            self.filters *= 2

        self.avg_pool = GlobalAveragePooling2D()
        self.f = Dense(class_number, activation='softmax')

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.blocks(x)
        x = self.avg_pool(x)
        y = self.f(x)

        return y


# res50、res101、res152 class
class Deep_ResBlock(Model):
    def __init__(self, filters, strides=1, residual_dimension=False):
        super(Deep_ResBlock, self).__init__()
        self.filters = filters
        self.residual_dimension = residual_dimension

        self.conv1 = Conv2D(filters, (1, 1), strides=strides)
        self.conv2 = Conv2D(filters, (3, 3), strides=1, padding='same')
        self.conv3 = Conv2D(4 * filters, (1, 1), strides=1)
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.relu1 = Activation('relu')
        self.relu2 = Activation('relu')
        self.relu3 = Activation('relu')

        '''
            每个残差块的第一块输入x通道数由64 -> 256  128 -> 512 256 -> 1024 512 -> 2048(4倍)
            可以理解为块中输入x的通道数与第三个卷积层输出通道数保持一致
            两个不同的残差块中，后一个的输出通道数为前一个的两倍，输出的宽高为前一个的一半。
        '''
        if residual_dimension:
            self.residual_conv = Conv2D(4 * filters, (1, 1), strides=strides)
            self.residual_bn = BatchNormalization()

    def call(self, inputs):
        residual_x = inputs

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        Fx = self.bn3(x)

        if self.residual_dimension:
            residual_x = self.residual_conv(residual_x)
            residual_x = self.residual_bn(residual_x)

        output = self.relu3(Fx + residual_x)
        return output


class Deep_Res(Model):
    def __init__(self, blocks_list, class_number):
        super(Deep_Res, self).__init__()
        self.blocks_num = len(blocks_list)
        self.blocks_list = blocks_list

        self.conv1 = Conv2D(64, (7, 7), strides=2)
        self.bn = BatchNormalization()
        self.relu = Activation('relu')
        self.max_pool = MaxPool2D((3, 3), strides=2)

        self.blocks = Sequential()
        self.filters = 64
        for block_id in range(self.blocks_num):  # 第几个块
            for layer_id in range(self.blocks_list[block_id]):  # 块里第几个
                if block_id == 0 and layer_id == 0:
                    block = Deep_ResBlock(self.filters, residual_dimension=True)
                elif block_id != 0 and layer_id == 0:
                    block = Deep_ResBlock(self.filters, strides=2, residual_dimension=True)
                else:
                    block = Deep_ResBlock(self.filters)
                self.blocks.add(block)
            self.filters *= 2

        self.avg_pool = GlobalAveragePooling2D()
        self.f = Dense(class_number, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.blocks(x)
        x = self.avg_pool(x)
        y = self.f(x)

        return y

