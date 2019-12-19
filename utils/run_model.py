import argparse
import os
import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
import struct
import cv2

# my classes + functions
from utils.bounding_box import BoundingBox
from utils.read_weights import ReadWeights
from utils.util_extras import *
from utils.read_image_data import read_image_data

def run_model(input_shape, output):
    pass;
    """
    model= Sequential()
    model.add(Conv2D(kernel_size=(3,3), filters=32, activation='tanh', input_shape=input_shape, use_bias=True, kernel_regularizer=ks.regularizers.l1_l2(l1=0.01, l2=0.01)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(kernel_size=(3,3), filters=64, activation='tanh'))
    model.add(MaxPool2D(pool_size=(3,3)))
    
    model.add(Flatten())
    
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=output, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
    
    return model

    # correct the sizes of the bounding boxes
    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

    # suppress non-maximal boxes
    do_nms(boxes, nms_thresh)     

    # draw bounding boxes on the image using labels
    draw_boxes(image, boxes, labels, obj_thresh) 
 
    # write the image with bounding boxes to file
    cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], (image).astype('uint8')) 
    """
    