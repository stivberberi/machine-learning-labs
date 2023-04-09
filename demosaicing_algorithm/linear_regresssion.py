import numpy as np


def generate_patch(size: int, image: np.ndarray, x: int, y: int):
    """Generates a patch of size x around the pixel at (x, y).

    Args:
        size (int): Size of the patch.
        image (np.ndarray): Image to generate the patch from.
        x (int): X coordinate of the pixel.
        y (int): Y coordinate of the pixel.

    Returns:
        np.ndarray: Patch of size x around the pixel at (x, y).
    """
    patch = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            patch[i, j] = image[x - size//2 + i, y - size//2 + j]
    return patch


def generate_training_data(images: list):
    """Generates the training data for the linear regression model.

    Args:
        images (list): List of training images.

    Returns:
        tuple: Tuple of 8 np arrays containing the x and y matrices for each patch type.
    """

    # Define all X matrices (8) as empty 2d np matrices
    # naming scheme is x_<pixel> where pixel is the color of the pixel
    # ex. for green, is either x_gb or x_gr depending on the row (blue or red row)
    x_gb, x_gr, x_r, x_b = np.empty((1, 25)), np.empty(
        (1, 25)), np.empty((1, 25)), np.empty((1, 25))

    # Same for y matrices
    y_gb, y_gr, y_r, y_b = np.empty((1, 1)), np.empty(
        (1, 1)), np.empty((1, 1)), np.empty((1, 1))

    for image in images:
        print(image.shape)
        # loop through x,y coords but leave 2 pixels on each side for padding
        for i in range(2, image.shape[0] - 2):
            for j in range(2, image.shape[1] - 2):
                # generate patches of size 5x5
                patch = generate_patch(5, image, i, j)
                flat_patch = np.reshape(patch, (1, 25))
                print(flat_patch.shape)

                # 4 patch cases based on pixel location and rggb bayer pattern
                if i % 2 == 0 and j % 2 == 0:
                    # both even, green pixel in blue row
                    np.append(x_gb, flat_patch, axis=0)
                    np.append(y_gb, image[i, j, 1], axis=0)

                    print(x_gb.shape)
                    print(y_gb.shape)
                    return

                if i % 2 == 1 and j % 2 == 1:
                    # both odd, green pixel in red row
                    np.append(x_gr, patch.flatten())
                    np.append(y_gr, image[i, j])

                if i % 2 == 0 and j % 2 == 1:
                    # even row, odd column, blue pixel
                    np.append(x_b, patch.flatten())
                    np.append(y_b, image[i, j])

                if i % 2 == 1 and j % 2 == 0:
                    # odd row, even column, red pixel
                    np.append(x_r, patch.flatten())
                    np.append(y_r, image[i, j])

    print('hi')
    print(x_b.shape)
    return x_gb, x_gr, x_r, x_b, y_gb, y_gr, y_r, y_b


def train_model(images: list):
    """Trains a linear regression model using the training images.

    Args:
        images (list): List of training images.

    Returns:
        tuple: Tuple of 8 np arrays containing the coefficients for each patch type.
    """
    # generate training data
    x_gb, x_gr, x_r, x_b, y_gb, y_gr, y_r, y_b = generate_training_data(images)

    # get coefficient matrices (8) using linear regression
    # naming scheme is a_<pixel>_<source> where pixel is the colour it is trying to predict and source is the X matrix
    # green predictor coefficients
    a_g_r = np.matmul(np.linalg.inv(np.matmul(x_r.T, x_r)),
                      np.matmul(x_r.T, y_r[:, :, 1]))
    a_g_r = np.linalg.lstsq(x_r, y_r[:, :, 1], rcond=None)[0]
    a_g_b = np.linalg.lstsq(x_b, y_b[:, :, 1], rcond=None)[0]

    # blue predictor coefficients
    a_b_gb = np.linalg.lstsq(x_gb, y_gb[:, :, 2], rcond=None)[0]
    a_b_gr = np.linalg.lstsq(x_gr, y_gr[:, :, 2], rcond=None)[0]
    a_b_r = np.linalg.lstsq(x_r, y_r[:, :, 2], rcond=None)[0]

    # red predictor coefficients
    a_r_gb = np.linalg.lstsq(x_gb, y_gb[:, :, 0], rcond=None)[0]
    a_r_gr = np.linalg.lstsq(x_gr, y_gr[:, :, 0], rcond=None)[0]
    a_r_b = np.linalg.lstsq(x_b, y_b[:, :, 0], rcond=None)[0]

    # write coefficients as columns in a csv file with variable as header
    with open('coefficients.csv', 'w') as f:
        f.write('a_g_r,a_g_b,a_b_gb,a_b_gr,a_b_r,a_r_gb,a_r_gr,a_r_b\n')
        for i in range(25):
            f.write(
                f'{a_g_r[i]},{a_g_b[i]},{a_b_gb[i]},{a_b_gr[i]},{a_b_r[i]},{a_r_gb[i]},{a_r_gr[i]},{a_r_b[i]}\n')

    model = (a_g_r, a_g_b, a_b_gb, a_b_gr, a_b_r, a_r_gb, a_r_gr, a_r_b)
    return model
