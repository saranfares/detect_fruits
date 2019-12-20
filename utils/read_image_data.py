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


from utils.bounding_box import BoundingBox
from utils.util_extras import convert_to_yolo

JPG_EXT = ".jpg"
XML_EXT = ".xml"

BLACK = 0
WHITE = 255

TRAIN_IMG_PATH = "pictures/train"
TEST_IMG_PATH = "pictures/test"


INDICES = {"banana":0, "orange":1, "apple":2, "mixed":3}
LAB_MULT = [0,0,0,1]

def read_image_data(path, current_type, desired_shape):
    print("Currently working on the", current_type, "image set")
    print("The images are in the folder", path)
    train_images_names = os.listdir(path)
    img_names = []

    # "boxes" ->  list of BoundingBox objects
    # "images" -> list of np arr of imgs

    labels_list = []
    images = []
    boxes_list = []
    shape = (desired_shape[0],desired_shape[1])
    for image in train_images_names:
        # files and locations
        img_name = image.split(".")
        img_name = img_name[0]
        if img_name not in img_names and (img_name is not ""):
            xml_file = path + "/" + img_name + XML_EXT
            img_file = path + "/" + img_name + JPG_EXT
            # read in image
            img_og = io.imread(img_file)
            img = cv2.resize(img_og,shape)
            img = img.astype('float32')
            #img /= 255.0
            images.append(img[:,:,0:3])
            shape_old = img_og.shape
            # read in location on the image
            img_info = ET.parse(xml_file)
            root = img_info.getroot()

            boxes = {"label":[], "bounds":[]}
            for obj in root.iter('object'):
                for item in obj:
                    if item.tag == "name":
                        boxes['label'].append(item.text)
                    elif item.tag == "pose":
                        pass;
                    elif item.tag == "bndbox":
                        boundaries = {}
                        for info_on_location in item:
                             boundaries[info_on_location.tag] = info_on_location.text
                        boxes['bounds'].append(boundaries)

            bx_l = []
            lab = [0,0,0,0]
            for idx in range(0,len(boxes["label"])):
                label_txt = boxes["label"][idx]
                lab[INDICES[label_txt]] = 1
                boundaries = boxes["bounds"][idx]
                x1 = boundaries["xmin"]
                y1 = boundaries["ymin"]
                x2 = boundaries["xmax"]
                y2 = boundaries["ymax"]
                bx = convert_to_yolo(shape_old, float(x1), float(y1), float(x2), float(y2))
                bx_l.append(bx)
            if sum(lab) is not 1:
                lab[3] = 1
            labels_list.append(lab)
            boxes_list.append(bx_l)
            img_names.append(img_name)

    print("Done working on the", current_type, "image set\n")
    return(img_names, images, boxes_list, labels_list)

""""
!!!! EXAMPLE OF XML FILE !!!!
    <size>
        <width>350</width>
        <height>350</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>apple</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>10</xmin>
            <ymin>8</ymin>
            <xmax>344</xmax>
            <ymax>336</ymax>
        </bndbox>
    </object>
"""

def main():
    train_images_names, train_images = read_image_data(TRAIN_IMG_PATH, "training")
    test_images_names, test_images = read_image_data(TEST_IMG_PATH, "test")   

if __name__ == "__main__":
    main()


