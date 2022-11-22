import pandas as pd
import numpy as np
import logging
from sklearn import tree, model_selection
import math

'''
Stiv Berberi
berbers
400212350
Lab 4
'''

STUDENT_SEED = 2350


def setup_logger():
    """Sets up the root logger to save all data for this assignment
    """
    # setup logger for writing to file
    logging.basicConfig(filename='lab_4.log', filemode='w',
                        level=logging.INFO, format='%(message)s')
    # add a stream handler to also send the output to stdout
    logging.getLogger().addHandler(logging.StreamHandler())


def load_data():
    """Loads the data from the csv file and returns it as a numpy array after shuffling.

    Returns:
        numpy.array: X and t paramater and target matrixes
    """
    dataset = pd.read_csv('data/spambase.data', header=None)
    X = dataset.iloc[:, :-1].values
    t = dataset.iloc[:, -1].values

    np.random.seed(STUDENT_SEED)
    np.random.shuffle(X)
    np.random.seed(STUDENT_SEED)
    np.random.shuffle(t)

    return X, t


def split_data(X, t, test_size=0.33):
    """Splits the data into training and test sets

    Args:
        X (numpy.array): X parameter matrix
        t (numpy.array): t target matrix
        test_size (float, optional): Size of the test set. Defaults to 0.33.

    Returns:
        tuple: X_train, X_test, t_train, t_test
    """

    X_test, X_train = np.split(X, [int(len(X) * test_size)])
    t_test, t_train = np.split(t, [int(len(t) * test_size)])

    # # Add a column of ones to the beginning of each X matrix
    # X_train = np.insert(X_train, 0, 1, axis=1)
    # X_test = np.insert(X_test, 0, 1, axis=1)

    return X_train, X_test, t_train, t_test


def calc_dt_cross_validation_error(X, t, num_leaves):
    """Performs a k-fold cross validation algorithm on given X and t matrices

    Args:
        X: Feature matrix
        t: Target matrix
        num_leaves: Number of leaves to use for the decision tree

    Returns:
        float: Average error of the cross-validation.
    """
    num_k_folds = 5
    # list of indexes for k-fold cross validation
    kf = model_selection.KFold(n_splits=num_k_folds)
    total_error = 0
    for train_index, test_index in kf.split(X):
        # training and test indexes for each iteration of cross validation.
        X_train, X_test = X[train_index], X[test_index]
        t_train, t_test = t[train_index], t[test_index]

        # train the model
        model = tree.DecisionTreeClassifier(max_leaf_nodes=num_leaves)
        model.fit(X_train, t_train)

        y = model.predict(X_test)

        # get the test error using the w parameters made from the training data.
        error = sum((y-t_test)**2) / len(t_test)
        total_error += error

    avg_error = total_error / num_k_folds

    return avg_error


def generate_decision_tree_classifier(X, t):
    """Trains a decision tree classifier on the training data and returns the model. Performs cross
    validation error calculation to find the best maximum number of leaves between 2 and 400.

    Args:
        X (numpy.array): X parameter matrix
        t (numpy.array): t target matrix

    Returns:
        sklearn.tree.DecisionTreeClassifier: Trained decision tree classifier
    """
    best_error = (math.inf, 2)
    for max_leaves in range(2, 400):
        # get the cross validation error for the current max_leaves
        error = calc_dt_cross_validation_error(X, t, max_leaves)
        if error < best_error[0]:
            best_error = (error, max_leaves)

    logging.info(f'Best error: {best_error[0]} at {best_error[1]} leaves')

    # train the model with the best max_leaves
    model = tree.DecisionTreeClassifier(max_leaf_nodes=best_error[1])
    model.fit(X, t)

    return model


def get_bagging_classifiers(X, t):


def main():
    """Main function
    """
    setup_logger()
    X, t = load_data()
    X_train, X_test, t_train, t_test = split_data(X, t)

    dt_model = generate_decision_tree_classifier(X_train, t_train)


main()
