import argparse
import os
import numpy as np
import struct
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras as kr

# my classes + functions
from utils.bounding_box import BoundingBox
from utils.util_extras import *
from utils.read_image_data import read_image_data
from utils.create_model import create_model
from utils.train_model import train_model
from utils.test_model import test_model

model_name = "test_latest.h5"

def run_model(train_path, test_path, output_path, input_shape, output):
    # create model
    # run the model on unseen input (from test set)
    print("loading test images into memory")
    test_image_names, test_images, test_boxes, test_labels = read_image_data(test_path, "test", input_shape)   

    try:
        print("attempting to load the pre-trained model into memory")
        trained_model = kr.models.load_model(model_name)
        print("loaded the pre-trained model into memory")

    except:
        print("unable to find pre-trained model...")
        print("now, building the model from scratch")
        num_classes = 3
        #anchors = [[30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

        model = create_model(input_shape, output, [], num_classes)

        # load image sets
        print("loading training images into memory")
        train_image_names, train_images, train_boxes, train_labels = read_image_data(train_path, "training", input_shape)

        # train the model
        trained_model = train_model(model, train_image_names, train_images, train_boxes, train_labels, test_images, test_labels)

        # save the model
        print("save model")
        trained_model.save(model_name)


    # test the model
    print("running model on unseen images")
    result_labels = test_model(trained_model, test_image_names, test_images, test_boxes, test_labels)
    return result_labels

    #print("done testing. images can be found in ", output_path)

    """
    # correct the sizes of the bounding boxes
    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

    # suppress non-maximal boxes
    do_nms(boxes, nms_thresh)     

    # draw bounding boxes on the image using labels
    draw_boxes(image, boxes, labels, obj_thresh) 
 
    # write the image with bounding boxes to file
    cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], (image).astype('uint8')) 
    """
