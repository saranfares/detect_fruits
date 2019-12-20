import keras as ks
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from keras.models import Sequential

from utils.util_extras import convolve_block_creator


def create_model(input_shape, output, anchors, num_classes):
    model= Sequential()
    model.add(Conv2D(kernel_size=(3,3), filters=32, activation='tanh', input_shape=input_shape, use_bias=True, kernel_regularizer=ks.regularizers.l1_l2(l1=0.001, l2=0.002)))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(kernel_size=(3,3), filters=64, activation='tanh',use_bias=True, kernel_regularizer=ks.regularizers.l1_l2(l1=0.001, l2=0.002)))
    model.add(Conv2D(kernel_size=(3,3), filters=64, activation='tanh',use_bias=True, kernel_regularizer=ks.regularizers.l1_l2(l1=0.001, l2=0.002)))
    model.add(MaxPool2D(pool_size=(3,3)))

    model.add(Conv2D(kernel_size=(3,3), filters=64, activation='tanh',use_bias=True, kernel_regularizer=ks.regularizers.l1_l2(l1=0.001, l2=0.002)))
    model.add(MaxPool2D(pool_size=(3,3)))

    model.add(Conv2D(kernel_size=(3,3), filters=64, activation='tanh',use_bias=True, kernel_regularizer=ks.regularizers.l1_l2(l1=0.001, l2=0.002)))
    model.add(MaxPool2D(pool_size=(3,3)))

    model.add(Flatten())
    
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=output, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()

    return model
