import numpy as np


def generate_X_m(M: int, x_train):
    '''
    Code to generate matrixes for M=0 to M=9 dimensions.
    Based on the M inputted, the x_train matrix will be repeatedly raised to a power from 1 to M and
    appended to the end of a ones column.
    '''
    X_m = np.ones((len(x_train), 1))    # column of ones
    x_train = np.array([x_train])     # add [] to make 2D array (transposable)
    for m in range(1, M+1):
        x_train_m = np.power(x_train.T, m)
        X_m = np.append(X_m, x_train_m, axis=1)

    return X_m


def train_w()


def main():
    X_train = np.linspace(0., 1., 10)  # training set
    X_valid = np.linspace(0., 1., 100)  # validation set
    np.random.seed(2350)
    # sin function output with added noise
    t_valid = np.sin(4*np.pi*X_valid) + 0.3 * np.random.randn(100)
    t_train = np.sin(4*np.pi*X_train) + 0.3 * np.random.randn(10)

    '''
    Store each training matrix of size M + 1 into a dict
    '''
    X_m = {
        'M0': None,
        'M1': None,
        'M2': None,
        'M3': None,
        'M4': None,
        'M5': None,
        'M6': None,
        'M7': None,
        'M8': None,
        'M9': None,
    }
    i = 0
    for m in X_m:
        X_m[m] = generate_X_m(i, X_train)
        i += 1
    print(X_m)


main()
