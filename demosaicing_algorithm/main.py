from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from linear_interpolation import demosaic_linear_interpolation


def load_training_images():
    """Loads training images from the training folder.
    """
    # load training images of .tif format
    # images taken from McMaster dataset: https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm
    # images are 512x512 pixels with 3 channels (RGB)
    # image = Image.open('McM/1.tif')
    image = Image.open('test_images/test1.png')
    return image


def main():
    """Main function entry point.
    """
    image = load_training_images()
    img_array = np.array(image)

    # generate a demosaiced image using linear interpolation
    demosaiced_image = demosaic_linear_interpolation(img_array)

    # plot the original image and the filtered image
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(demosaiced_image)
    plt.show()

    return


if __name__ == '__main__':
    main()
