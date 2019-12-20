import argparse
import os
import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
import struct
import cv2

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class BoundingBox:
    def __init__(self, x1, y1, x2, y2, obj = None, lab = None):
        self.xmin = x1
        self.ymin = y1
        self.xmax = x2
        self.ymax = y2
        
        self.obj = obj
        self.labs = lab

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.labs)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.labs[self.get_label()]
            
        return self.score

