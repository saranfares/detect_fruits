import sys

# our internal run_model function
from utils.run_model import run_model
from utils.show_images import show_images


OUTPUT_PATH = "pictures/output/"
INPUT_TRAIN = "pictures/train/"
INPUT_TEST = "pictures/test/"
LIMIT = 8
IMG_SHAPE = (128,128,1)

def main(print_imgs, boxes):
    # run run_model 
    input_shape = IMG_SHAPE
    labels = ["apple", "banana", "orange", "multiple"]
    output_sz = len(labels)

    result_labels = run_model(INPUT_TRAIN, INPUT_TEST, OUTPUT_PATH, input_shape, output_sz)

    if print_imgs:
        print("Running print images portion")
        # open a window and display the images next to the original
        if boxes:
            show_images(INPUT_TEST, OUTPUT_PATH, result_labels, LIMIT, boxes=boxes)
        else:
            show_images(INPUT_TEST, OUTPUT_PATH, result_labels, LIMIT, boxes=boxes)

if __name__ == '__main__':
    print_imgs = True
    boxes = False
    main(print_imgs, boxes)
