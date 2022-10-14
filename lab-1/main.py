import numpy as np
from matplotlib import pyplot as plt


class LinearRegressionModelling():
    def __init__(self, seed) -> None:
        '''
        Initialise training and validation sets
        '''
        self.X_train = np.linspace(0., 1., 10)  # training set
        self.X_valid = np.linspace(0., 1., 100)  # validation set
        np.random.seed(seed)
        # sin function output with added noise
        self.t_valid = np.sin(4*np.pi*self.X_valid) + \
            0.3 * np.random.randn(100)
        self.t_train = np.sin(4*np.pi*self.X_train) + 0.3 * np.random.randn(10)

        '''
        Store each training matrix and corresponding weights into a dict
        '''
        self.models = {
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

    def train_model(self):
        '''
        Train linear models for M=0 to M=9
        '''
        i = 0
        for m in self.models:
            self.models[m]['X_m'] = self.generate_X_m(i, self.X_train)
            i += 1
            self.models[m]['w'] = self.train_w(
                self.models[m]['X_m'], self.t_train)

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

    def generate_model_plot(model, X_valid, t_train, t_valid):
        '''
        Generates a plot for a model, calculating (and plotting) the training and validation error as well
        '''
        fig = plt.figure()
        # 111 to specify that 1x1 subplots are on the graph
        axis = fig.add_subplot(111)

        f_true_x = np.arange(0, 10, 0.1)
        f_true_y = np.sin(f_true_x)
        # True plot will be in red
        axis.scatter(f_true_x, f_true_y, c='r', label='f_true')

        plt.show()


def main():
    '''
    Main function that uses the linear model to generate plots
    '''
    lm = LinearRegressionModelling(seed=2350)


main()
