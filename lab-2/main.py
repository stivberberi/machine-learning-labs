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


def calc_cross_validation_error(X, t):
    """Performs a k-fold cross validation algorithm on given X and t matrices

    Args:
        X: Feature matrix
        t: Target matrix

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

        w = linear_regression(X_train, t_train)
        y = np.matmul(X_test, w)

        # get the test error using the w parameters made from the training data.
        error = sum((y-t_test)**2) / len(t_test)
        total_error += error

    avg_error = total_error / num_k_folds

    return avg_error


def calc_test_error(X_train, t_train, X_test, t_test):
    """Calculate the test error of a given linear model

    Args:
        X_train: Paramaters matrix for training set
        t_train: Target vector for training set
        X_test: Paramaters matrix for test set
        t_test: Target vector for test set

    Returns:
        float: Calculated test error
    """

    w = linear_regression(X_train, t_train)

    y = np.matmul(X_test, w)
    error = sum((y-t_test)**2) / len(t_test)

    return error


def get_basis_cross_validation_error(basis_exponent, X_train, X_test, t_train):
    # do a basis expansion with a given exponent
    X_train = np.column_stack(
        (X_train, X_train[:, -1]**basis_exponent))
    X_test = np.column_stack(
        (X_test, X_test[:, -1]**basis_exponent))

    basis_cv_error = calc_cross_validation_error(X_train, t_train)

    return X_train, X_test, basis_cv_error


def run_basis_expansion(X_train, X_test, t_train, t_test, cross_validation_error):
    """Performs a basis expansion at least twice

    Args:
        X_train: Paramaters matrix for training set
        t_train: Target vector for training set
        X_test: Paramaters matrix for test set
        t_test: Target vector for test set
        cross_validation_error: Error from the unexpanded set
        last_feature: Last added feature to the set.

    Returns:
        _type_: _description_
    """

    '''
    Run 2 basis expansions with exponents 2 and 0.5 first.
    If these don't  
    '''
    best_basis = 0
    best_error = 1000000000     # very large error

    basis_x_train, basis_x_test, first_basis_cv_error = get_basis_cross_validation_error(
        2, X_train, X_test, t_train)
    logging.info(
        f'\nCV error for basis expansion of exponent 2 is: {first_basis_cv_error}')

    basis_x_train, basis_x_test, second_basis_cv_error = get_basis_cross_validation_error(
        0.5, X_train, X_test, t_train)
    logging.info(
        f'CV error for basis expansion of exponent 0.5 is: {second_basis_cv_error}')

    if first_basis_cv_error < second_basis_cv_error:
        best_basis = 2
    else:
        best_basis = 0.5

    logging.info(f'The best chosen basis was exponent of {best_basis}')

    # get the test error of the best basis
    basis_test_error = calc_test_error(
        basis_x_train, t_train, basis_x_test, t_test)

    logging.info(
        f'Test error for basis of exponent {best_basis} is: {basis_test_error}')

    return best_error, basis_test_error


def main():
    setup_logger()
    X_test, X_train, t_test, t_train = get_data_sets()

    # set of available parameters
    features_available = list(range(0, 13))

    # list of sets to be filled in... S1, S2, etc.
    current_set_X_train = np.ones(len(X_train))
    current_set_X_test = np.ones(len(X_test))
    sets = []

    # list of errors for plotting
    test_errors = []

    for k in range(0, 13):
        min_error = 10000000        # very large error to set as the initial minimum error
        selected_feature = -1
        logging.info(f'\nCreating set {k+1}')
        # loop through the number of parameters
        for feature in features_available:
            temp_X = np.column_stack(
                (current_set_X_train, X_train[:, feature]))
            error = calc_cross_validation_error(temp_X, t_train)
            logging.info(
                f'Feature {feature+1} for set {k+1}-> Error = {error}')

            # keep the lowest error of all paramaters
            if error < min_error:
                selected_feature = feature
                min_error = error

        # now have selected the best feature; remove it from the set
        features_available.remove(selected_feature)
        logging.info(f'Added feature {selected_feature+1} to set {k+1}')

        # update the current Set
        current_set_X_train = np.column_stack(
            (current_set_X_train, X_train[:, selected_feature]))
        current_set_X_test = np.column_stack(
            (current_set_X_test, X_test[:, selected_feature]))

        # calculate test error for the given model, save it to a list
        test_error = calc_test_error(
            current_set_X_train, t_train, current_set_X_test, t_test)
        logging.info(f'Test error for set {k+1} is: {test_error}')
        test_errors.append(test_error)

        basis_error, basis_test_error = run_basis_expansion(
            current_set_X_train, current_set_X_test, t_train, t_test, min_error)

    return


main()
