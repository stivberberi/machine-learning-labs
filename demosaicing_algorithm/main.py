from PIL import Image
import numpy as np
import os
from linear_regresssion import train_model
from utils.generate_patches import generate_mosaic_patch_greyscale
from utils.demosaic import demosaic_image, load_model_from_csv


def load_training_images():
    """Loads training images from the training folder.
    """
    # load training images of .tif format
    # images taken from McMaster dataset: https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm
    # images are 512x512 pixels with 3 channels (RGB)
    training_images_dir = 'training_images/'
    images = []
    for filename in os.listdir(training_images_dir):
        if filename.endswith('.tif'):
            images.append(np.array(Image.open(training_images_dir + filename)))

    return images


def main():
    """Main function entry point.
    """

    # train model
    # training_images = load_training_images()
    # train_model(training_images)

    # load model from csv file
    model = load_model_from_csv()

    # demosaic the image
    test_image = Image.open('test_images/test2.png')
    test_image = np.array(test_image)
    demosaic_img = demosaic_image(test_image, model)

    # save the image
    Image.fromarray(demosaic_img.astype('uint8')).save(
        'output_images/test2_demosaic.png')

    return


if __name__ == '__main__':
    main()
