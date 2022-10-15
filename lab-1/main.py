from cProfile import label
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
            self.models[m]['w'] = self.train_w(self.models[m]['X_m'], 
                                               self.t_train)

    def generate_X_m(self, M: int, X):
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

    def train_w(self, X, t):
        '''
        Calculate the weights when training a set of examples for a given target set
        '''
        t = np.array([t]).T
        XTX = np.dot(X.T, X)
        if np.linalg.det(XTX) == 0:
            num_rows, num_col = np.shape(X)  # get the number of examples
            w = np.zeros((num_rows, 1))
        else:
            a = np.linalg.inv(XTX)
            b = np.dot(X.T, t)
            w = np.dot(a,b)

        return w

    def calc_error(self, X, w, t):
        '''
        Calculate the error (training or validation) based on given training data, weights,
        and a target set (training or validation)
        '''
        t = np.array([t]).T
        num_rows, num_col = np.shape(X)  # get the number of examples
        Xw = np.dot(X, w)
        error = (1 / num_rows) * np.dot((Xw - t).T, (Xw - t))

        return error

    def generate_model_plot(self, m: int):
        '''
        Generates a plot for a model, calculating (and plotting) the training and validation error as well
        '''
        # m is used to act as a unique identifier for the figure so we don't add data to the same plot every time
        fig = plt.figure(m)
        # 111 to specify that 1x1 subplots are on the graph
        axis = fig.add_subplot(111)
        f_true_x = np.arange(0, 1, 0.01)
        f_true_y = np.sin(4*np.pi*f_true_x)
        # True plot will be in red
        axis.plot(f_true_x, f_true_y, 'r', label='f_true')

        # plot validation set as blue
        axis.plot(self.X_valid, self.t_valid,
                  'bo', label='Validation Set')
        # plot validation set
        axis.plot(self.X_train, self.t_train,
                  'go', label='Training Set')

        # plot predicted function; get the model by the m index (convert to list so we can index by number)
        model_index = list(self.models)[m]
        X_m = self.models[model_index]['X_m']
        w = self.models[model_index]['w']
        f_predict = np.dot(X_m, w)
        axis.plot(self.X_train, f_predict, 'y', label='Predicted output')

        plt.title(f'Hyperparamater M{m} Plot')
        plt.xlabel('X')
        plt.ylabel('f(x)')
        plt.legend(loc='upper right')

        fig.savefig(f'figures/M{m}.png')


def main():
    '''
    Main function that uses the linear model to generate plots
    '''
    lm = LinearRegressionModelling(seed=2350)
    lm.train_model()
    for i in range(len(lm.models)):
        lm.generate_model_plot(m=i)

    # Get training and validation error
    error_fig = plt.figure('error')
    axis = error_fig.add_subplot(111)
    x = [0,1,2,3,4,5,6,7,8,9]
    train_err = []
    valid_err = []

    i = 0
    for model in lm.models:
        err_t = lm.calc_error(X=lm.models[model]['X_m'],
                            w=lm.models[model]['w'],
                            t=lm.t_train)
        print(err_t)
        # X_m_val = lm.generate_X_m(M=i, X=lm.X_valid)
        # err_v = lm.calc_error(X=X_m_val,
        #                       w=lm.models[model]['w'],
        #                       t=lm.t_valid)
        # valid_err.append(err_v)
        train_err.append(err_t[0][0])

    axis.plot(x, train_err, 'r', label='Training Error')
    axis.plot(x, valid_err, 'b', label='Validation Error')
    plt.title('Training and Validation Error')
    plt.xlabel('M')
    plt.ylabel('Error (Mean squared error)')
    plt.legend(loc='upper right')
    error_fig.savefig('figures/ErrorPlot.png')


main()
