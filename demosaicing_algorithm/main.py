from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from linear_interpolation import demosaic_linear_interpolation
from linear_regresssion import train_model, generate_patch


def load_training_images():
    """Loads training images from the training folder.
    """
    # load training images of .tif format
    # images taken from McMaster dataset: https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm
    # images are 512x512 pixels with 3 channels (RGB)
    # image = Image.open('McM/1.tif')
    image = Image.open('McM/1.tif')
    return image


def demosaic_image(image, model):
    """Demosaics the image using coeeficients from the linear regression model.
    """

    # expand model matrixes
    a_g_r, a_g_b, a_b_gb, a_b_gr, a_b_r, a_r_gb, a_r_gr, a_r_b = model

    # initiate array of same shape as image but with 3 channels
    demosaic_img = np.zeros((image.shape[0], image.shape[1], 3))
    for i in range(2, image.shape[0] - 2):
        for j in range(2, image.shape[1] - 2):
            # generate patches of size 5x5
            patch = generate_patch(5, image, i, j)

            # 4 patch cases based on pixel location and rggb bayer pattern
            if i % 2 == 0 and j % 2 == 0:
                # both even, green pixel in blue row, predict the red and blue channels
                demosaic_image[i, j, 0] = np.dot(a_r_gb, patch.flatten())
                demosaic_img[i, j, 1] = image[i, j]
                demosaic_img[i, j, 2] = np.dot(a_b_gb, patch.flatten())

            if i % 2 == 1 and j % 2 == 1:
                # both odd, green pixel in red row, predict the red and blue channels
                demosaic_image[i, j, 0] = np.dot(a_r_gr, patch.flatten())
                demosaic_img[i, j, 1] = image[i, j]
                demosaic_img[i, j, 2] = np.dot(a_b_gr, patch.flatten())

            if i % 2 == 0 and j % 2 == 1:
                # blue pixel, predict the green and blue channels
                demosaic_img[i, j, 0] = np.dot(a_r_b, patch.flatten())
                demosaic_img[i, j, 1] = np.dot(a_g_b, patch.flatten())
                demosaic_image[i, j, 3] = image[i, j]

            if i % 2 == 1 and j % 2 == 0:
                # red pixel, predict the green and red channels
                demosaic_img[i, j, 0] = image[i, j]
                demosaic_img[i, j, 1] = np.dot(a_g_r, patch.flatten())
                demosaic_image[i, j, 2] = np.dot(a_b_r, patch.flatten())

    return demosaic_img


def load_model_from_csv():
    """Loads the linear regression model from a csv file.
    """

    # load model from csv file
    model = np.loadtxt('model.csv', delimiter=',')
    return model


def main():
    """Main function entry point.
    """
    training_images = load_training_images()
    img_array = np.array(training_images)

    # train model
    model = train_model([img_array])

    # demosaic the image
    test_image = Image.open('test_images/test1.png')
    demosaic_img = demosaic_image(test_image, model)

    # plot the original image and the filtered image
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(test_image, cmap='gray')
    ax[1].imshow(demosaic_image)
    plt.show()

    return


if __name__ == '__main__':
    main()
