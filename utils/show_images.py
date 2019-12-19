# import the necessary packages
import os
import sys

# the packages to read in xml files (image loc data)
import xml.etree.ElementTree as ET

import numpy as np
import imutils
import time
import cv2
import skimage.io as io


def show_images(input_path, output_path):
    print("showing images")
    names = os.listdir(input_path)
    img_names = set()

    for image in names:
        # files and locations
        img_name = image.split(".")
        img_name = img_name[0]
        if img_name not in img_names:
            img_names.add(img_name)
    for img_name in img_names:
        # find test image original
        og_f = input_path + img_name + '.jpg'
        # find output matching one
        w_boxes_f = output_path + img_name + '_boxes.jpg'
        # display side-by-side
        