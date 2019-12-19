import sys

# our internal run_model function
from utils.run_model import run_model
from utils.show_images import show_images


OUTPUT_PATH = "pictures/output/"
INPUT_TRAIN = "pictures/train/"
INPUT_TEST = "pictures/test/"


def main(print_imgs):
    if print_imgs:
        # open a window and display the images next to the original
        show_images(INPUT_TEST, OUTPUT_PATH)

if __name__ == '__main__':
    print_imgs = sys.argv[1]
    main(print_imgs)
