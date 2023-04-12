from generate_patches import generate_mosaic_patch_greyscale


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
            patch = generate_mosaic_patch_greyscale(image, i, j)
            flat_patch = np.reshape(patch, (1, 25))

            if i % 2 == 0 and j % 2 == 0:
                # both even, red pixel
                demosaic_img[i, j, 0] = image[i, j]
                demosaic_img[i, j, 1] = np.dot(flat_patch, a_g_r)
                demosaic_img[i, j, 2] = np.dot(flat_patch, a_b_r)

            if i % 2 == 0 and j % 2 == 1:
                # even x, odd y, green pixel in red row
                demosaic_img[i, j, 0] = np.dot(flat_patch, a_r_gr)
                demosaic_img[i, j, 1] = image[i, j]
                demosaic_img[i, j, 2] = np.dot(flat_patch, a_b_gr)

            if i % 2 == 1 and j % 2 == 0:
                # odd x, even y, green pixel in blue row
                demosaic_img[i, j, 0] = np.dot(flat_patch, a_r_gb)
                demosaic_img[i, j, 1] = image[i, j]
                demosaic_img[i, j, 2] = np.dot(flat_patch, a_b_gb)

            if i % 2 == 1 and j % 2 == 1:
                # both odd, blue pixel
                demosaic_img[i, j, 0] = np.dot(flat_patch, a_r_b)
                demosaic_img[i, j, 1] = np.dot(flat_patch, a_g_b)
                demosaic_img[i, j, 2] = image[i, j]

    return demosaic_img


def extract_numeric(cell):
    """Extracts numeric values from a string. Removes all the [] characters.
    """
    return float(cell.strip('[]'))


def load_model_from_csv():
    """Loads the linear regression model from a csv file.
    """

    # load model from csv file
    data = np.loadtxt('coefficients.csv', delimiter=',', skiprows=1, dtype=None, encoding=None, converters={
                      i: extract_numeric for i in range(8)})

    # convert each column to a 25x1 matrix
    a_g_r = data[:, 0].reshape(25, 1)
    a_g_b = data[:, 1].reshape(25, 1)
    a_b_gb = data[:, 2].reshape(25, 1)
    a_b_gr = data[:, 3].reshape(25, 1)
    a_b_r = data[:, 4].reshape(25, 1)
    a_r_gb = data[:, 5].reshape(25, 1)
    a_r_gr = data[:, 6].reshape(25, 1)
    a_r_b = data[:, 7].reshape(25, 1)

    model = a_g_r, a_g_b, a_b_gb, a_b_gr, a_b_r, a_r_gb, a_r_gr, a_r_b
    return model
