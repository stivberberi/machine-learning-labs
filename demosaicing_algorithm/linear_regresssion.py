from utils.generate_patches import generate_mosaic_patch_rgb
import numpy as np
from datetime import datetime


def generate_training_data(images: list):
    """Generates the training data for the linear regression model.

    Args:
        images (list): List of training images.

    Returns:
        tuple: Tuple of 8 np arrays containing the x and y matrices for each patch type.
    """

    # Define all X matrices (8) as empty 2d np matrices
    # naming scheme is x_<pixel> where pixel is the color of the pixel
    x_gb, x_gr, x_r, x_b = np.empty((1, 25)), np.empty(
        (1, 25)), np.empty((1, 25)), np.empty((1, 25))

    # t_<target_colour>_<current_pixel>
    # ex. t_r_gb is the target red value for a pixel in the green/blue row
    t_g_r, t_g_b, t_r_gb, t_r_gr, t_r_b, t_b_gb, t_b_gr, t_b_r = np.empty((1, 1)), np.empty((1, 1)), np.empty(
        (1, 1)), np.empty((1, 1)), np.empty((1, 1)), np.empty((1, 1)), np.empty((1, 1)), np.empty((1, 1))

    for count, image in enumerate(images):
        print(
            f'Generating training data for image {count + 1} of {len(images)} [{datetime.now().strftime("%H:%M:%S")}]')
        # loop through x,y coords but leave 2 pixels on each side for padding
        for x in range(2, image.shape[0] - 2):
            for y in range(2, image.shape[1] - 2):
                # generate patches of size 5x5
                patch = generate_mosaic_patch_rgb(image, x, y)
                flat_patch = np.reshape(patch, (1, 25))

                if x % 2 == 0 and y % 2 == 0:
                    # both even, red pixel
                    x_r = np.vstack((x_r, flat_patch))
                    t_g_r = np.vstack((t_g_r, np.array([[image[x, y, 1]]])))
                    t_b_r = np.vstack((t_b_r, np.array([[image[x, y, 2]]])))

                if x % 2 == 0 and y % 2 == 1:
                    # even x, odd y, green pixel in red row
                    x_gr = np.vstack((x_gr, flat_patch))
                    t_r_gr = np.vstack((t_r_gr, np.array([[image[x, y, 0]]])))
                    t_b_gr = np.vstack((t_b_gr, np.array([[image[x, y, 2]]])))

                if x % 2 == 1 and y % 2 == 0:
                    # odd x, even y, green pixel in blue row
                    x_gb = np.vstack((x_gb, flat_patch))
                    t_r_gb = np.vstack((t_r_gb, np.array([[image[x, y, 0]]])))
                    t_b_gb = np.vstack((t_b_gb, np.array([[image[x, y, 2]]])))

                if x % 2 == 1 and y % 2 == 1:
                    # both odd, blue pixel
                    x_b = np.vstack((x_b, flat_patch))
                    t_r_b = np.vstack((t_r_b, np.array([[image[x, y, 0]]])))
                    t_g_b = np.vstack((t_g_b, np.array([[image[x, y, 1]]])))

    # remove the first row of zeros
    x_gb, x_gr, x_r, x_b = x_gb[1:, :], x_gr[1:, :], x_r[1:, :], x_b[1:, :]
    t_r_b, t_r_gb, t_r_gr, t_g_b, t_g_r, t_b_gb, t_b_gr, t_b_r = t_r_b[1:, :], t_r_gb[1:,
                                                                                      :], t_r_gr[1:, :], t_g_b[1:, :], t_g_r[1:, :], t_b_gb[1:, :], t_b_gr[1:, :], t_b_r[1:, :]

    return x_gb, x_gr, x_r, x_b, t_g_b, t_g_r, t_r_b, t_r_gb, t_r_gr, t_b_gb, t_b_gr, t_b_r


def train_model(images: list):
    """Trains a linear regression model using the training images.

    Args:
        images (list): List of training images.

    Returns:
        tuple: Tuple of 8 np arrays containing the coefficients for each patch type.
    """
    # generate training data
    x_gb, x_gr, x_r, x_b, t_g_b, t_g_r, t_r_b, t_r_gb, t_r_gr, t_b_gb, t_b_gr, t_b_r = generate_training_data(
        images)

    # get coefficient matrices (8) using linear regression
    # naming scheme is a_<pixel>_<source> where pixel is the colour it is trying to predict and source is the X matrix
    print('Training green coefficients...')
    a_g_r = np.matmul(np.matmul(np.linalg.inv(
        np.matmul(x_r.T, x_r)), x_r.T), t_g_r)
    a_g_b = np.matmul(np.matmul(np.linalg.inv(
        np.matmul(x_b.T, x_b)), x_b.T), t_g_b)

    print('Training blue coefficients...')
    a_b_gb = np.matmul(np.matmul(np.linalg.inv(
        np.matmul(x_gb.T, x_gb)), x_gb.T), t_b_gb)
    a_b_gr = np.matmul(np.matmul(np.linalg.inv(
        np.matmul(x_gr.T, x_gr)), x_gr.T), t_b_gr)
    a_b_r = np.matmul(np.matmul(np.linalg.inv(
        np.matmul(x_r.T, x_r)), x_r.T), t_b_r)

    print('Training red coefficients...')
    a_r_gb = np.matmul(np.matmul(np.linalg.inv(
        np.matmul(x_gb.T, x_gb)), x_gb.T), t_r_gb)
    a_r_gr = np.matmul(np.matmul(np.linalg.inv(
        np.matmul(x_gr.T, x_gr)), x_gr.T), t_r_gr)
    a_r_b = np.matmul(np.matmul(np.linalg.inv(
        np.matmul(x_b.T, x_b)), x_b.T), t_r_b)

    # write coefficients as columns in a csv file with variable as header
    with open('coefficients1.csv', 'w') as f:
        f.write('a_g_r,a_g_b,a_b_gb,a_b_gr,a_b_r,a_r_gb,a_r_gr,a_r_b\n')
        for i in range(25):
            f.write(
                f'{a_g_r[i]},{a_g_b[i]},{a_b_gb[i]},{a_b_gr[i]},{a_b_r[i]},{a_r_gb[i]},{a_r_gr[i]},{a_r_b[i]}\n')

    return
