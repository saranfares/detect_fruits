import numpy as np
import keras as ks

from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from keras.models import Sequential

from tensorflow.keras.models import load_model

def train_model(model_skeleton, names, images, boxes, labels):
    print("training the model")
    labels = np.array(labels)
    shape = (240,128,128,1)
    images = np.array(images)
    images = np.reshape(images, shape)
    print(len(labels))
    tracking_training = model_skeleton.fit(images, labels, batch_size=240, epochs=200)
    print(tracking_training)
    return model_skeleton
