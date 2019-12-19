import sys

# our internal run_model function
from utils.run_model import run_model
from utils.show_images import show_images


OUTPUT_PATH = "pictures/output/"
INPUT_TRAIN = "pictures/train/"
INPUT_TEST = "pictures/test/"
LIMIT = 2
IMG_SHAPE = (128,128)

def main(print_imgs):
    # run run_model 
    input_shape = IMG_SHAPE
    labels = ["apple", "orange", "banana"]
    output_sz = len(labels)
    run_model(input_shape, output_sz)

    if print_imgs:
        # open a window and display the images next to the original
        show_images(INPUT_TEST, OUTPUT_PATH, LIMIT)

if __name__ == '__main__':
    print_imgs = False
    main(print_imgs)
