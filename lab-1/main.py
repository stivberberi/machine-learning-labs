import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import os


def init_X_valid_and_t_valid(seed):
    '''
    Initialise validation set
    '''
    X_train = np.linspace(0., 1., 10)  # training set
    X_valid = np.linspace(0., 1., 100)  # validation set

    np.random.seed(seed)
    # sin function output with added noise
    t_valid = np.sin(4*np.pi*X_valid) + 0.3 * np.random.randn(100)
    np.random.seed(seed)
    t_train = np.sin(4*np.pi*X_train) + 0.3 * np.random.randn(10)

    return X_train, X_valid, t_train, t_valid


def generate_X_m(M: int, X):
    '''
    Code to generate matrixes for M=0 to M=9 dimensions.
    Based on the M inputted, the x_train matrix will be repeatedly raised to a power from 1 to M and
    appended to the end of a ones column.
    '''
    X_m = np.ones((len(X), 1))    # column of ones
    # add [] to make 2D array (so it's transposable)
    X = np.array([X])
    for m in range(1, M+1):
        x_train_m = np.power(X.T, m)
        X_m = np.append(X_m, x_train_m, axis=1)

    return X_m


def train_w(X_m, t):
    '''
    Code to train the weights for each model.
    '''
    w = np.linalg.solve(np.dot(X_m.T, X_m), np.dot(X_m.T, t))
    return w


def train_w_regularised(X_m, t, l):
    '''
    Code to train the weights for model with regularisation.
    '''
    w = np.linalg.solve(np.dot(X_m.T, X_m) + l *
                        np.identity(X_m.shape[1]), np.dot(X_m.T, t))
    return w


def plot_model(X_valid, t_valid, X_train, t_train, M, w, X_m):
    '''
    Code to plot the model for each M. Also plots the f true function.
    '''
    # m is used to act as a unique identifier for the figure so we don't add data to the same plot every time
    fig = plt.figure(M)
    # 111 to specify that 1x1 subplots are on the graph
    axis = fig.add_subplot(111)
    f_true_x = np.arange(0, 1, 0.01)
    f_true_y = np.sin(4*np.pi*f_true_x)
    # True plot will be in red
    axis.plot(f_true_x, f_true_y, 'r', label='f_true')

    # plot validation set as blue
    axis.plot(X_valid, t_valid, 'bo', label='Validation Set')
    # plot validation set
    axis.plot(X_train, t_train, 'go', label='Training Set')

    # plot the predicted function
    axis.plot(X_train, np.dot(X_m, w), 'y', label='Predicted Function')

    axis.set_title(f'Model M={M}')
    axis.set_xlabel('x')
    axis.set_ylabel('f(x)')
    axis.legend()

    fig.savefig(f'figures/M{M}.png')


def calc_training_error(X_m, w, t_train):
    '''
    Code to calculate the training error for each model.
    '''
    training_error = 1/X_m.shape[0] * \
        np.sum(np.power(np.dot(X_m, w) - t_train, 2))
    return training_error


def calc_validation_error(X_m, w, t_valid):
    '''
    Code to calculate the validation error for each model.
    '''
    validation_error = 1/X_m.shape[0] * \
        np.sum(np.power(np.dot(X_m, w) - t_valid, 2))
    return validation_error


def main():
    SEED = 2350

    X_train, X_valid, t_train, t_valid = init_X_valid_and_t_valid(seed=SEED)

    if not os.path.exists('figures'):
        os.makedirs('figures')
    else:
        # remove all files in figures folder
        for file in os.listdir('figures'):
            os.remove(f'figures/{file}')

    training_error = []
    validation_error = []
    for m in range(10):
        X_m_train = generate_X_m(m, X_train)
        X_m_valid = generate_X_m(m, X_valid)
        w = train_w(X_m_train, t_train)
        plot_model(X_valid, t_valid, X_train, t_train, m, w, X_m_train)

        training_error.append(calc_training_error(X_m_train, w, t_train))
        validation_error.append(calc_validation_error(X_m_valid, w, t_valid))

    # average error between all x_valid points and f_true
    f_true_x = np.arange(0, 1, 0.01)
    f_true_y = np.sin(4*np.pi*f_true_x)
    average_error = 1/len(f_true_x) * \
        np.sum(np.power(f_true_y - t_valid, 2))

    # plot training and validation error
    fig = plt.figure('Training and Validation Error')
    axis = fig.add_subplot(111)
    axis.plot(training_error, 'r', label='Training Error')
    axis.plot(validation_error, 'b', label='Validation Error')
    axis.plot([average_error]*10, 'g',
              label='Average Error between targets and true function')
    axis.set_title('Training and Validation Error')
    axis.set_xlabel('M')
    axis.set_ylabel('Error')
    axis.legend()
    fig.savefig('figures/Training and Validation Error.png')

    '''----Regularisation----'''

    X_m_train = generate_X_m(9, X_train)
    X_m_valid = generate_X_m(9, X_valid)
    # remove the first column of X_m_train and X_m_valid
    X_m_train = np.delete(X_m_train, 0, 1)
    X_m_valid = np.delete(X_m_valid, 0, 1)

    # standardize features
    sc = StandardScaler()
    XX_train = sc.fit_transform(X_m_train)
    XX_valid = sc.transform(X_m_valid)

    # get 20 lambdas between 0 and 1 inclusive
    lambdas = np.linspace(0, 1, 1000)
    training_error_regularised = []
    validation_error_regularised = []
    for l in lambdas:
        w = train_w_regularised(XX_train, t_train, l)
        training_error_regularised.append(calc_training_error(
            XX_train, w, t_train) + l * np.sum(np.power(w, 2)))
        validation_error_regularised.append(calc_validation_error(
            XX_valid, w, t_valid) + l * np.sum(np.power(w, 2)))

    # plot training and validation error for regularised model
    fig = plt.figure('Training and Validation Error for Regularised Model')
    axis = fig.add_subplot(111)
    axis.plot(lambdas, training_error_regularised, 'r', label='Training Error')
    axis.plot(lambdas, validation_error_regularised,
              'b', label='Validation Error')
    axis.set_title('Training and Validation Error for Regularised Model')
    axis.set_xlabel('Lambda')
    axis.set_ylabel('Error')
    axis.legend()
    fig.savefig(
        'figures/Training and Validation Error for Regularised Model.png')

    # train regularised model with lambda = 0.000000001
    w = train_w_regularised(XX_train, t_train, 0.000000001)
    plot_model(X_valid, t_valid, X_train, t_train,
               'Regularised with 0.000000001', w, X_m_train)

    # train regularised model with lambda = 0.001
    w = train_w_regularised(XX_train, t_train, 0.001)
    plot_model(X_valid, t_valid, X_train, t_train,
               'Regularised with 0.001', w, X_m_train)


if __name__ == "__main__":
    main()
