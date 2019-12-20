import keras as ks
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from keras.models import Sequential

import numpy as np

INDICES = {"banana":0, "orange":1, "apple":2, "mixed":3}
labels_strings = ["banana", "orange", "apple", "mixed"]

def test_model(model, names, images, boxes, labels):
    print("testing on a few images... the image along with output from the model will be shown. ")
    labs = []
    for idx in range(0,len(images)):
        shape = (1,128,128,1)
        img = np.array(images[idx])
        img = np.reshape(img, shape)
        lab = model.predict_classes(img)
        print(lab[0])
        labs.append(labels_strings[lab[0]])

    return labs
