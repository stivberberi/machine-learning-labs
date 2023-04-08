from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff


def load_training_images():
    """Loads training images from the training folder.
    """
    # load training images of .tif format
    # images taken from McMaster dataset: https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm
    # images are 512x512 pixels

    image = tiff.imread('McM/1.tif')
    return image


def main():
    """Main function entry point.
    """
    image = load_training_images()
    img_array = np.array(image)

    return


if __name__ == '__main__':
    main()
