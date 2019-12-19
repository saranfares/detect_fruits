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
 

JPG_EXT = ".jpg"
XML_EXT = ".xml"

BLACK = 0
WHITE = 255

TRAIN_IMG_PATH = "pictures/train"
TEST_IMG_PATH = "pictures/test"


def read_image_data(path, current_type):
    print("Currently working on the", current_type, "image set")
    print("The images are in the folder", path)
    train_images_names = os.listdir(path)
    img_names = set()

    # these will be stored as follows: IN A DICT with keys that are == to image_name
    # image_name : "label" -> [] labels for each of the objects in img (can be multiple)
    #                       -- stored as an array
    #              "bounds"-> [] bounds for each of the objs labeled in "labels"
    #                       -- stored as an array
    #                           -- array contains a dict that has:
    #                           --> "xmin", "xmax", "ymin", "ymax" as keys
    #              "image_matrix" -> np arr of img
    #              "size" --> stores a dict of info
    #                       --> "width", "height", "depth" as keys
    #

    train_images = {}
    for image in train_images_names:
        # files and locations
        img_name = image.split(".")
        img_name = img_name[0]
        if img_name not in img_names and (img_name is not ""):
            xml_file = path + "/" + img_name + XML_EXT
            img_file = path + "/" + img_name + JPG_EXT
            train_images[img_name] = {}
            # read in image
            img = io.imread(img_file, as_gray=True)
            train_images[img_name]["image_matrix"] = img
            # read in location on the image
            img_info = ET.parse(xml_file)
            root = img_info.getroot()
            train_images[img_name]['size'] = {}
            for obj in root.iter('size'):
                for size_info in obj:
                    train_images[img_name]['size'][size_info.tag] = size_info.text
            train_images[img_name]['label'] = []
            train_images[img_name]['bounds'] = []
            for obj in root.iter('object'):
                for item in obj:
                    if item.tag == "name":
                        train_images[img_name]['label'].append(item.text)
                    elif item.tag == "pose":
                        pass;
                    elif item.tag == "bndbox":
                        boundaries = {}
                        for info_on_location in item:
                             boundaries[info_on_location.tag] = info_on_location.text
                        train_images[img_name]['bounds'].append(boundaries)

                # we want to find 1. size , 2. object info
                #print(size.tag, size.attrib)
            img_names.add(img_name)
    print("Done working on the", current_type, "image set\n")
    return(img_names, train_images)

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


