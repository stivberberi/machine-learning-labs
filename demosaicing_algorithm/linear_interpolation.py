import numpy as np
from scipy.signal import convolve2d


def demosaic_linear_interpolation(raw_img: np.ndarray):
    """Demosaics an image using linear interpolation.

    Args:
        raw_img (np.ndarray): Raw image numpy array to be demosaiced.

    Returns:
        np.ndarray: Demosaiced image as a numpy array.
    """
    image_width, image_height = raw_img.shape

    # creating masks for the bayer pattern
    bayer_red = np.tile(np.array([[1, 0], [0, 0]]), (np.ceil(
        image_width/2).astype(int), np.ceil(image_height/2).astype(int)))
    bayer_blue = np.tile(np.array([[0, 0], [0, 1]]), (np.ceil(
        image_width/2).astype(int), np.ceil(image_height/2).astype(int)))
    bayer_green = np.tile(np.array([[0, 1], [1, 0]]), (np.ceil(
        image_width/2).astype(int), np.ceil(image_height/2).astype(int)))

    # truncating the extra pixels at the edges
    if (image_width % 2) == 1:
        bayer_red = np.delete(bayer_red, -1, axis=0)
        bayer_blue = np.delete(bayer_blue, -1, axis=0)
        bayer_green = np.delete(bayer_green, -1, axis=0)
    if (image_height % 2) == 1:
        bayer_red = np.delete(bayer_red, -1, axis=1)
        bayer_blue = np.delete(bayer_blue, -1, axis=1)
        bayer_green = np.delete(bayer_green, -1, axis=1)

    # extracting the red, green and blue components of the image using the mask
    red_image = raw_img * bayer_red
    blue_image = raw_img * bayer_blue
    green_image = raw_img * bayer_green

    # deducing the green pixels at missing points
    green = green_image + convolve2d(green_image, np.array(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]])/4, mode='same', boundary='symm')

    # deducing the red pixels at missing points
    red_1 = convolve2d(red_image, np.array(
        [[1, 0, 1], [0, 0, 0], [1, 0, 1]])/4, mode='same', boundary='symm')
    red_2 = convolve2d(red_image, np.array(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]])/2, mode='same', boundary='symm')
    red = red_image + red_1 + red_2

    # deducing the blue pixels at missing points
    blue_1 = convolve2d(blue_image, np.array(
        [[1, 0, 1], [0, 0, 0], [1, 0, 1]])/4, mode='same', boundary='symm')
    blue_2 = convolve2d(blue_image, np.array(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]])/2, mode='same', boundary='symm')
    blue = blue_image + blue_1 + blue_2

    image = np.zeros((image_width, image_height, 3))
    image[:, :, 0] = red
    image[:, :, 1] = green
    image[:, :, 2] = blue

    return image.astype(np.uint8)
