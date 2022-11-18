import numpy as np
from sklearn.datasets import load_breast_cancer
import logging

'''
Stiv Berberi
berbers
400212350
Lab 3
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
    """Loads breast cancer data set, shuffling the examples first.

    Returns:
        X,t (tuple): Parameter and target data sets
    """
    data = load_breast_cancer()
    X = data.data
    t = data.target
    # shuffle the data
    np.random.seed(STUDENT_SEED)
    np.random.shuffle(X)
    np.random.seed(STUDENT_SEED)
    np.random.shuffle(t)

    return X, t


def get_data_sets():
    """Separates breast cancer data set to training (75%) and validation (test) sets.

    Returns:
        X_test, X_train, T_test, T_train: Training and validation data sets
    """
    X, t = load_data()

    X_test, X_train = np.split(X, [int(len(X) * 0.25)])
    T_test, T_train = np.split(t, [int(len(t) * 0.25)])

    return X_test, X_train, T_test, T_train


def vectorized_logistic_regression(X, t):
    """Performs vectorized logistic regression on the given data set.

    Args:
        X (np.ndarray): Data set
        t (np.ndarray): Target data set

    Returns:
        w (np.ndarray): Weight vector
    """

    XT = np.transpose(X)
    XTX = np.matmul(XT, X)
    # compute w if XTX is non-singular (det != 0)
    if np.linalg.det(XTX):
        w = np.matmul(np.matmul(np.linalg.inv(XTX), XT), t)
    else:
        w = np.zeros(X[0].shape)

    z = np.matmul(X, w)


def main():
    data = load_data()


if __name__ == "__main__":
    main()
