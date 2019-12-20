# import the necessary packages
import os
import sys

import numpy as np
import matplotlib.pyplot as plt 
import skimage.io as io

ROWS = 1
COLS = 2

def show_images(input_path, output_path, labels, limit, boxes):
    print("showing images")
    names = os.listdir(input_path)
    img_names = set()

    for image in names:
        # files and locations
        img_name = image.split(".")
        img_name = img_name[0]
        if img_name not in img_names:
            if len(img_name)>=1:
                img_names.add(img_name)

    count = 1
    for img_name in img_names:
        lab = labels[count-1]
        print(img_name)
        # find test image original
        og_f = input_path + img_name + '.jpg'
        # find output matching one
        w_boxes_f = output_path + img_name + '_boxes.jpg'

        label = img_name.split("_")[0]
        if label == "mixed":
            label = "multiple fruits"

        img_og = io.imread(og_f)

        

        if boxes:
            # display side-by-side
            plt.subplot(ROWS, COLS, 1)
            plt.imshow(img_og, cmap=plt.cm.gray)
            plt.title("Test Image")
            plt.xticks([])
            plt.yticks([])

            img_boxes = io.imread(w_boxes_f)
            plt.subplot(ROWS, COLS, 2)
            plt.imshow(img_boxes, cmap=plt.cm.gray)
            plt.title("Boxed + Labeled Test Image")
            plt.xticks([])
            plt.yticks([])
        else:
            plt.imshow(img_og, cmap=plt.cm.gray)
            plt.title("Image was labeled as "+ lab)
            plt.xticks([])
            plt.yticks([])


        plt.show()
        io.show()

        # show only 3 examples
        if count >= limit:
            break;
        count += 1