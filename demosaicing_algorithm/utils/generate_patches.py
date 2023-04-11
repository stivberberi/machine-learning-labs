import numpy as np

# Define the standard patch size
PATCH_SIZE = 5


def _rggb(image: np.ndarray, x: int, y: int):
    patch = np.zeros((PATCH_SIZE, PATCH_SIZE))

    for i in range(PATCH_SIZE):
        for j in range(PATCH_SIZE):
            if i % 2 == 0 and j % 2 == 0:
                patch[i, j] = image[x - PATCH_SIZE//2 + i,
                                    y - PATCH_SIZE//2 + j, 0]        # red
            if i % 2 == 0 and j % 2 == 1:
                patch[i, j] = image[x - PATCH_SIZE//2 + i,
                                    y - PATCH_SIZE//2 + j, 1]        # green
            if i % 2 == 1 and j % 2 == 0:
                patch[i, j] = image[x - PATCH_SIZE//2 + i,
                                    y - PATCH_SIZE//2 + j, 1]        # green
            if i % 2 == 1 and j % 2 == 1:
                patch[i, j] = image[x - PATCH_SIZE//2 + i,
                                    y - PATCH_SIZE//2 + j, 2]        # blue

    return patch


def _gbrg(image: np.ndarray, x: int, y: int):
    patch = np.zeros((PATCH_SIZE, PATCH_SIZE))

    for i in range(PATCH_SIZE):
        for j in range(PATCH_SIZE):
            if i % 2 == 0 and j % 2 == 0:
                patch[i, j] = image[x - PATCH_SIZE//2 + i,
                                    y - PATCH_SIZE//2 + j, 1]        # green
            if i % 2 == 0 and j % 2 == 1:
                patch[i, j] = image[x - PATCH_SIZE//2 + i,
                                    y - PATCH_SIZE//2 + j, 2]        # blue
            if i % 2 == 1 and j % 2 == 0:
                patch[i, j] = image[x - PATCH_SIZE//2 + i,
                                    y - PATCH_SIZE//2 + j, 0]        # red
            if i % 2 == 1 and j % 2 == 1:
                patch[i, j] = image[x - PATCH_SIZE//2 + i,
                                    y - PATCH_SIZE//2 + j, 1]        # green

    return patch


def _grbg(image: np.ndarray, x: int, y: int):
    patch = np.zeros((PATCH_SIZE, PATCH_SIZE))

    for i in range(PATCH_SIZE):
        for j in range(PATCH_SIZE):
            if i % 2 == 0 and j % 2 == 0:
                patch[i, j] = image[x - PATCH_SIZE//2 + i,
                                    y - PATCH_SIZE//2 + j, 1]        # green
            if i % 2 == 0 and j % 2 == 1:
                patch[i, j] = image[x - PATCH_SIZE//2 + i,
                                    y - PATCH_SIZE//2 + j, 0]        # red
            if i % 2 == 1 and j % 2 == 0:
                patch[i, j] = image[x - PATCH_SIZE//2 + i,
                                    y - PATCH_SIZE//2 + j, 2]        # blue
            if i % 2 == 1 and j % 2 == 1:
                patch[i, j] = image[x - PATCH_SIZE//2 + i,
                                    y - PATCH_SIZE//2 + j, 1]        # green

    return patch


def _bggr(image: np.ndarray, x: int, y: int):
    patch = np.zeros((PATCH_SIZE, PATCH_SIZE))

    for i in range(PATCH_SIZE):
        for j in range(PATCH_SIZE):
            if i % 2 == 0 and j % 2 == 0:
                patch[i, j] = image[x - PATCH_SIZE//2 + i,
                                    y - PATCH_SIZE//2 + j, 2]        # blue
            if i % 2 == 0 and j % 2 == 1:
                patch[i, j] = image[x - PATCH_SIZE//2 + i,
                                    y - PATCH_SIZE//2 + j, 1]        # green
            if i % 2 == 1 and j % 2 == 0:
                patch[i, j] = image[x - PATCH_SIZE//2 + i,
                                    y - PATCH_SIZE//2 + j, 1]        # green
            if i % 2 == 1 and j % 2 == 1:
                patch[i, j] = image[x - PATCH_SIZE//2 + i,
                                    y - PATCH_SIZE//2 + j, 0]        # red

    return patch


def generate_mosaic_patch_rgb(image: np.ndarray, x: int, y: int):
    """Generates a patch of size 'PATCH_SIZE' around the pixel at (x, y) based on the x and y coordinates.

    Args:
        size (int): Size of the patch.
        image (np.ndarray): Image to generate the patch from.
        x (int): X coordinate of the pixel.
        y (int): Y coordinate of the pixel.

    Returns:
        np.ndarray: Patch of size 'PATCH_SIZE' around the pixel at (x, y).
    """

    if x % 2 == 0 and y % 2 == 0:
        patch = _rggb(image, x, y)
    elif x % 2 == 0 and y % 2 == 1:
        patch = _grbg(image, x, y)
    elif x % 2 == 1 and y % 2 == 0:
        patch = _gbrg(image, x, y)
    elif x % 2 == 1 and y % 2 == 1:
        patch = _bggr(image, x, y)

    return patch


def generate_mosaic_patch_greyscale(image: np.ndarray, x: int, y: int):
    """Generates a patch of size 'PATCH_SIZE' around the pixel at (x, y) based on the x and y coordinates.

    Args:
        size (int): Size of the patch.
        image (np.ndarray): Image to generate the patch from.
        x (int): X coordinate of the pixel.
        y (int): Y coordinate of the pixel.

    Returns:
        np.ndarray: Patch of size 'PATCH_SIZE' around the pixel at (x, y).
    """

    patch = np.zeros((PATCH_SIZE, PATCH_SIZE))

    for i in range(PATCH_SIZE):
        for j in range(PATCH_SIZE):
            patch[i, j] = image[x - PATCH_SIZE//2 + i,
                                y - PATCH_SIZE//2 + j]

    return patch
