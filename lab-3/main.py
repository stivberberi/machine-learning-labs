import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
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
    logging.basicConfig(filename='lab_3.log', filemode='w',
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
    """Separates breast cancer data set to training (75%) and validation (test) sets. Also performs feature normalization.

    Returns:
        X_test, X_train, T_test, T_train: Training and validation data sets
    """
    X, t = load_data()
    # split into 75% training and 25% test sets
    X_test, X_train = np.split(X, [int(len(X) * 0.25)])
    T_test, T_train = np.split(t, [int(len(t) * 0.25)])

    # normalize features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_test, X_train, T_test, T_train


def vectorized_logistic_regression(X, t) -> np.array:
    """Performs vectorized logistic regression on the given data set.

    Args:
        X (np.ndarray): Data set
        t (np.ndarray): Target data set

    Returns:
        z (np.ndarray): z vector
    """

    XT = np.transpose(X)
    XTX = np.matmul(XT, X)
    # compute w if XTX is non-singular (det != 0)
    if np.linalg.det(XTX):
        w = np.matmul(np.matmul(np.linalg.inv(XTX), XT), t)
    else:
        w = np.zeros(X[0].shape)

    # compute z vector = wT*X
    wT = np.transpose(w)
    z = np.matmul(X, wT)

    return z


def main():
    X_test, X_train, t_test, t_train = get_data_sets()
    # add ones column to X matrixes
    train_ones = np.ones(len(X_train))
    test_ones = np.ones(len(X_train))
    X_train = np.column_stack()

    print(
        f'xtrain: {X_train.shape}\t xtest {X_test.shape}\t ttrain {t_train.shape}\t ttest {t_test.shape}')
    z = vectorized_logistic_regression(X_train, t_train)


if __name__ == "__main__":
    main()
