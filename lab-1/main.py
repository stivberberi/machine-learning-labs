import numpy as np


def generate_X_m(M: int, x_train):
    '''
    Code to generate matrixes for M=0 to M=9 dimensions.
    Based on the M inputted, the x_train matrix will be repeatedly raised to a power from 1 to M and
    appended to the end of a ones column.
    '''
    X_m = np.ones((len(x_train), 1))    # column of ones
    # add [] to make 2D array (so it's transposable)
    x_train = np.array([x_train])
    for m in range(1, M+1):
        x_train_m = np.power(x_train.T, m)
        X_m = np.append(X_m, x_train_m, axis=1)

    return X_m


def train_w(X, t):
    '''
    Calculate the weights when training a set of examples for a given target set
    '''
    num_rows, num_col = np.shape(X)  # get the number of examples
    XTX = np.dot(X.T, X)
    if np.linalg.det(XTX) == 0:
        w = np.zeros((num_rows, 1))
    else:
        w = (np.linalg.inv(XTX)) * np.dot(X.T, t)

    return w


def calc_error(X, w, t):
    '''
    Calculate the error (training or validation) based on given training data, weights,
    and a target set (training or validation)
    '''
    num_rows, num_col = np.shape(X)  # get the number of examples
    error = (1 / num_col) * (X*w - t).T * (X*w - t)


def main():
    X_train = np.linspace(0., 1., 10)  # training set
    X_valid = np.linspace(0., 1., 100)  # validation set
    np.random.seed(2350)
    # sin function output with added noise
    t_valid = np.sin(4*np.pi*X_valid) + 0.3 * np.random.randn(100)
    t_train = np.sin(4*np.pi*X_train) + 0.3 * np.random.randn(10)

    '''
    Store each training matrix and corresponding weights into a dict
    '''
    models = {
        'M0': {'X_m': None, 'w': None},
        'M1': {'X_m': None, 'w': None},
        'M2': {'X_m': None, 'w': None},
        'M3': {'X_m': None, 'w': None},
        'M4': {'X_m': None, 'w': None},
        'M5': {'X_m': None, 'w': None},
        'M6': {'X_m': None, 'w': None},
        'M7': {'X_m': None, 'w': None},
        'M8': {'X_m': None, 'w': None},
        'M9': {'X_m': None, 'w': None},
    }
    i = 0
    for m in models:
        models[m]['X_m'] = generate_X_m(i, X_train)
        i += 1
        models[m]['w'] = train_w(models[m]['X_m'], t_train)


main()
