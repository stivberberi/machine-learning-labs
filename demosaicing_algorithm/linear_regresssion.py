import numpy as np
from scipy.ndimage import generic_filter


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

    # Define all X matrices (8)
    # naming scheme is x_<pixel> where pixel is the color of the pixel
    # ex. for green, is either x_gb or x_gr depending on the row (blue or red row)
    x_gb, x_gr, x_r, x_b = [], [], [], []

    # Same for y matrices
    y_gb, y_gr, y_r, y_b = [], [], [], []

    for image in images:
        # loop through x,y coords but leave 2 pixels on each side for padding
        for i in range(2, image.shape[0] - 2):
            for j in range(2, image.shape[1] - 2):
                # generate patches of size 5x5
                patch = generate_patch(5, image, i, j)

                # 4 patch cases based on pixel location and rggb bayer pattern
                if i % 2 == 0 and j % 2 == 0:
                    # both even, green pixel in blue row
                    x_gb.append(patch.flatten())
                    y_gb.append(image[i, j])

                if i % 2 == 1 and j % 2 == 1:
                    # both odd, green pixel in red row
                    x_gr.append(patch.flatten())
                    y_gr.append(image[i, j])

                if i % 2 == 0 and j % 2 == 1:
                    # even row, odd column, blue pixel
                    x_b.append(patch.flatten())
                    y_b.append(image[i, j])

                if i % 2 == 1 and j % 2 == 0:
                    # odd row, even column, red pixel
                    x_r.append(patch.flatten())
                    y_r.append(image[i, j])

    return x_gb, x_gr, x_r, x_b, y_gb, y_gr, y_r, y_b


def train_model(images: list):
    """Trains a linear regression model using the training images.

    Args:
        images (list): List of training images.

    Returns:
        LinearRegression: Linear regression model.
    """
    # generate training data
    x_gb, x_gr, x_r, x_b, y_gb, y_gr, y_r, y_b = generate_training_data(images)

    # get coefficient matrices (8) using linear regression
