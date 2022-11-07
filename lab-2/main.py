from sklearn import datasets, utils, model_selection
import numpy as np
import logging

'''
Stiv Berberi
berbers
400212350
Lab 2 
'''

STUDENT_SEED = 2350


def setup_logger():
    """Sets up the root logger to save all data for this assignment
    """
    # setup logger for writing to file
    logging.basicConfig(filename='lab_2.log', filemode='w',
                        level=logging.INFO, format='%(message)s')
    # add a stream handler to also send the output to stdout
    logging.getLogger().addHandler(logging.StreamHandler())


def load_data():
    """Loads boston housing data set, shuffling the examples first.

    Returns:
        X,t (tuple): Parameter and target data sets
    """
    data = datasets.load_boston()
    x = data.data
    t = data.target
    x, t = utils.shuffle(x, t, random_state=STUDENT_SEED)
    return x, t


def get_data_sets():
    """Separates boston housing data set to training (75%) and validation (test) sets.

    Returns:
        X_test, X_train, T_test, T_train: Training and validation data sets
    """
    X, t = load_data()

    X_test, X_train = np.split(X, [int(len(X) * 0.25)])
    T_test, T_train = np.split(t, [int(len(t) * 0.25)])

    return X_test, X_train, T_test, T_train


def linear_regression(X, t):
    """Generates linear regression model

    Args:
        x (numpy matrix): Feature matrix
        t (numpy matrix): Targets matrix

    Returns:
        w: Vector of paramaters
    """
    XT = np.transpose(X)
    XTX = np.matmul(XT, X)
    # compute w if XTX is non-singular (det != 0)
    if np.linalg.det(XTX):
        w = np.matmul(np.matmul(np.linalg.inv(XTX), XT), t)
    else:
        w = np.zeros(X[0].shape)
    return w


def calc_cross_validation_error(X):
    kf = model_selection.KFold()    # list of indexes for k-fold cross validation
    for train_index, test_index in kf.split()

def main():
    setup_logger()
    X_test, X_train, T_test, T_train = get_data_sets()

    current_X_train = np.ones(len(X_train))
    # set of available parameters
    features_available = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    # list of sets to be filled in... S1, S2, etc.
    sets = []
    for k in range(0, 13):
        logging.info(f'Creating set {k+1}')
        # loop through the number of parameters
        for feature in features_available:
            # goes in a set, so ordering will be random... O(1) lookup time however
            temp_X = np.column_stack((current_X_train, X_train[:, feature]))
            error = calc_cross_validation_error(temp_X)
    return


main()
