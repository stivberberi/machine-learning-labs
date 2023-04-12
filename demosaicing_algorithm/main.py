from PIL import Image
import numpy as np
import os
from linear_regresssion import train_model
from utils.generate_patches import generate_mosaic_patch_greyscale
from utils.demosaic import demosaic_image, load_model_from_csv
from linear_interpolation import demosaic_linear_interpolation


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


def calc_rmse(img1, img2):
    """Calculates the root mean squared error between two images.
    """
    return np.sqrt(np.mean((img1 - img2) ** 2))


def main():
    """Main function entry point.
    """

    # train model
    '''----Uncomment this to train the model!!---- '''

    # training_images = load_training_images()
    # train_model(training_images)

    # load model from csv file
    model = load_model_from_csv()

    '''
    RMSE calculations; each of the 5 images inside of 'rmse_images' folder will be 
    demosaiced using the regression model and linear interpolation
    The three greyscale and their RGB equivalents were taken from:
    https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/dataset/images.html
    '''
    # greyscale images
    img1 = np.array(Image.open('rmse_images/grey_1.jpg'))
    img2 = np.array(Image.open('rmse_images/grey_2.jpg'))
    img3 = np.array(Image.open('rmse_images/grey_3.jpg'))

    # RGB images
    img1_true = np.array(Image.open('rmse_images/rgb_1.jpg'))
    img2_true = np.array(Image.open('rmse_images/rgb_2.jpg'))
    img3_true = np.array(Image.open('rmse_images/rgb_3.jpg'))

    # demosaic images using regression model
    validation_1_regression = demosaic_image(img1, model)
    validation_2_regression = demosaic_image(img2, model)
    validation_3_regression = demosaic_image(img3, model)

    # Save them to the output_images folder
    Image.fromarray(validation_1_regression.astype('uint8')).save(
        'output_images/validation_1_regression.png')
    Image.fromarray(validation_2_regression.astype('uint8')).save(
        'output_images/validation_2_regression.png')
    Image.fromarray(validation_3_regression.astype('uint8')).save(
        'output_images/validation_3_regression.png')

    print(
        f'RMSE for image 1 - regression model: {calc_rmse(img1_true, validation_1_regression)}')
    print(
        f'RMSE for image 2 - regression model: {calc_rmse(img2_true, validation_2_regression)}')
    print(
        f'RMSE for image 3 - regression model: {calc_rmse(img3_true, validation_3_regression)}')

    validation_1_interpolation = demosaic_linear_interpolation(img1)
    validation_2_interpolation = demosaic_linear_interpolation(img2)
    validation_3_interpolation = demosaic_linear_interpolation(img3)

    # Save them to the output_images folder
    Image.fromarray(validation_1_interpolation.astype('uint8')).save(
        'output_images/validation_1_interpolation.png')
    Image.fromarray(validation_2_interpolation.astype('uint8')).save(
        'output_images/validation_2_interpolation.png')
    Image.fromarray(validation_3_interpolation.astype('uint8')).save(
        'output_images/validation_3_interpolation.png')

    print(
        f'RMSE for image 1 - linear interpolation: {calc_rmse(img1_true, validation_1_interpolation)}')
    print(
        f'RMSE for image 2 - linear interpolation: {calc_rmse(img2_true, validation_2_interpolation)}')
    print(
        f'RMSE for image 3 - linear interpolation: {calc_rmse(img3_true, validation_3_interpolation)}')

    # Calculate RMSE for the MATLAB generated images
    img1_matlab = np.array(Image.open('matlab_outputs/grey1_output.png'))
    img2_matlab = np.array(Image.open('matlab_outputs/grey2_output.png'))
    img3_matlab = np.array(Image.open('matlab_outputs/grey3_output.png'))

    print(
        f'RMSE for image 1 - MATLAB: {calc_rmse(img1_true, img1_matlab)}')
    print(
        f'RMSE for image 2 - MATLAB: {calc_rmse(img2_true, img2_matlab)}')
    print(
        f'RMSE for image 3 - MATLAB: {calc_rmse(img3_true, img3_matlab)}')

    # Finally, save the test image outputs from interpolation and regression
    test_image1 = np.array(Image.open('test_images/test1.png'))
    test_image2 = np.array(Image.open('test_images/test2.png'))

    # regression
    Image.fromarray(demosaic_image(test_image1, model).astype('uint8')).save(
        'output_images/test1_regression.png')
    Image.fromarray(demosaic_image(test_image2, model).astype('uint8')).save(
        'output_images/test2_regression.png')

    # interpolation
    Image.fromarray(demosaic_linear_interpolation(test_image1).astype('uint8')).save(
        'output_images/test1_interpolation.png')
    Image.fromarray(demosaic_linear_interpolation(test_image2).astype('uint8')).save(
        'output_images/test2_interpolation.png')

    return


if __name__ == '__main__':
    main()
