import numpy as np
import keras as ks

from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from keras.models import Sequential

from tensorflow.keras.models import load_model

def train_model(model_skeleton, names, images, boxes, labels, test_images, test_labels):
    print("training the model")
    labels = np.array(labels)
    imgs = np.zeros((len(labels),128,128,3))
    for i in range(0, imgs.shape[0]):
        imgs[i] = np.array(images[i])

    test_labels = np.array(test_labels)
    test_imgs = np.zeros((len(test_labels),128,128,3))
    for i in range(0, test_imgs.shape[0]):
        test_imgs[i] = np.array(test_images[i])

    tracking_training = model_skeleton.fit(imgs, labels, batch_size=30, epochs=20, validation_data=(test_imgs, test_labels))

    return model_skeleton
