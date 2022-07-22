from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Flatten, Dense, Dropout


class AlexNet8(Model):
    def __init__(self, class_number):
        super(AlexNet8, self).__init__()
        # 原文输入224*224*3，步长为4，我做的112*112，取步长为2
        self.c1 = Conv2D(filters=96, kernel_size=(11, 11), strides=2)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(2, 2))

        self.c2 = Conv2D(filters=256, kernel_size=(5, 5))
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(2, 2))

        self.c3 = Conv2D(filters=384, kernel_size=(3, 3), padding='same')
        self.c4 = Conv2D(filters=384, kernel_size=(3, 3), padding='same')
        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b3 = BatchNormalization()
        self.a3 = Activation('relu')
        self.p3 = MaxPool2D(pool_size=(2, 2))

        # 原论文用4096、4096、1000，修改为2048、2048、10
        self.flatten = Flatten()
        self.f1 = Dense(2048, activation='relu')
        self.d1 = Dropout(0.5)
        self.f2 = Dense(2048, activation='relu')
        self.d2 = Dropout(0.5)
        self.f3 = Dense(class_number, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)

        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.p3(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        y = self.f3(x)

        return y
