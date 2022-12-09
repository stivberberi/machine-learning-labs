from sklearn import tree, model_selection, ensemble, metrics
from multiprocessing import Process, Queue
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import logging
import math
import time

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
    logging.basicConfig(filename='test.log', filemode='w',
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


def calc_test_error(y, t):
    """Calculates the test error for the given y and t matrices

    Args:
        y (numpy.array): y matrix
        t (numpy.array): t matrix

    Returns:
        float: Test error
    """

    accuracy = metrics.accuracy_score(t, y)
    error = 1 - accuracy

    return error


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


def train_decision_tree_classifier(X, t):
    """Trains a decision tree classifier on the training data and returns the model. Performs cross
    validation error calculation to find the best maximum number of leaves between 2 and 400.

    Args:
        X (numpy.array): X parameter matrix
        t (numpy.array): t target matrix

    Returns:
        sklearn.tree.DecisionTreeClassifier: Trained decision tree classifier
        list: List of cross-validation errors for each number of leaves
    """
    best_error = (math.inf, 2)
    cv_errors = []
    for max_leaves in range(2, 400):
        # get the cross validation error for the current max_leaves
        error = calc_dt_cross_validation_error(X, t, max_leaves)
        cv_errors.append(error)
        if error < best_error[0]:
            best_error = (error, max_leaves)

    logging.info(f'Best error: {best_error[0]} at {best_error[1]} leaves')

    # train the model with the best max_leaves
    model = tree.DecisionTreeClassifier(max_leaf_nodes=best_error[1])
    model.fit(X, t)

    return model, cv_errors


def train_bagging_classifier(X, t, num_classifiers, X_test, t_test, queue):

    test_errors = queue.get()

    classifier = ensemble.BaggingClassifier(n_estimators=num_classifiers)
    classifier.fit(X, t)

    y = classifier.predict(X_test)
    error = calc_test_error(y, t_test)
    logging.info(
        f'Bagging test error: {error} at {num_classifiers} classifiers')
    test_errors.append(error)

    queue.put(test_errors)

    return


def train_random_forest_classifier(X, t, num_classifiers, X_test, t_test, queue):

    test_errors = queue.get()
    classifier = ensemble.RandomForestClassifier(n_estimators=num_classifiers)
    classifier.fit(X, t)

    y = classifier.predict(X_test)
    error = calc_test_error(y, t_test)
    logging.info(
        f'Random forest test error: {error} at {num_classifiers} classifiers')
    test_errors.append(error)

    queue.put(test_errors)

    return


def train_adaboost_classifier(X, t, num_classifiers, X_test, t_test, queue):

    test_errors = queue.get()
    # automatically initializes decision stump base classifier with depth 1.
    classifier = ensemble.AdaBoostClassifier(n_estimators=num_classifiers)
    classifier.fit(X, t)

    y = classifier.predict(X_test)
    error = calc_test_error(y, t_test)
    logging.info(
        f'Adaboost test error: {error} at {num_classifiers} classifiers')
    test_errors.append(error)

    queue.put(test_errors)

    return


def train_adaboost_classifier_with_depth_10(X, t, num_classifiers, X_test, t_test, queue):

    test_errors = queue.get()
    classifier = ensemble.AdaBoostClassifier(
        n_estimators=num_classifiers, base_estimator=tree.DecisionTreeClassifier(max_depth=10))
    classifier.fit(X, t)

    y = classifier.predict(X_test)
    error = calc_test_error(y, t_test)
    logging.info(
        f'Adaboost with depth 10 error: {error} at {num_classifiers} classifiers')
    test_errors.append(error)

    queue.put(test_errors)

    return


def train_adaboost_classifier_with_any_depth(X, t, num_classifiers, X_test, t_test, queue):

    test_errors = queue.get()
    classifier = ensemble.AdaBoostClassifier(
        n_estimators=num_classifiers, base_estimator=tree.DecisionTreeClassifier())
    classifier.fit(X, t)

    y = classifier.predict(X_test)
    error = calc_test_error(y, t_test)
    logging.info(
        f'Adaboost any depth error: {error} at {num_classifiers} classifiers')
    test_errors.append(error)

    queue.put(test_errors)

    return


def main_2():
    Bagging_test_errors = [0.06060606060606055, 0.059947299077733884, 0.054677206851119875, 0.05731225296442688, 0.054677206851119875, 0.05731225296442688, 0.059288537549407105, 0.058629776021080326, 0.058629776021080326, 0.05797101449275366, 0.05599472990777343, 0.05797101449275366, 0.058629776021080326, 0.0566534914361001, 0.05599472990777343, 0.05731225296442688, 0.05599472990777343, 0.05731225296442688, 0.05731225296442688, 0.0566534914361001, 0.0566534914361001, 0.05731225296442688, 0.05797101449275366, 0.0566534914361001,
                           0.05599472990777343, 0.0566534914361001, 0.05731225296442688, 0.0566534914361001, 0.05731225296442688, 0.0566534914361001, 0.05599472990777343, 0.0566534914361001, 0.05599472990777343, 0.059288537549407105, 0.058629776021080326, 0.0566534914361001, 0.0566534914361001, 0.0566534914361001, 0.0566534914361001, 0.0566534914361001, 0.05599472990777343, 0.05731225296442688, 0.0566534914361001, 0.05599472990777343, 0.059288537549407105, 0.05797101449275366, 0.0566534914361001, 0.0566534914361001, 0.0566534914361001, 0.05599472990777343]
    Random_forest_test_errors = [0.050065876152832645, 0.05270092226613965, 0.0487483530961792, 0.05204216073781287, 0.0513833992094862, 0.0487483530961792, 0.050065876152832645, 0.05204216073781287, 0.0487483530961792, 0.05204216073781287, 0.04940711462450598, 0.04808959156785242, 0.04940711462450598, 0.04808959156785242, 0.04808959156785242, 0.050065876152832645, 0.050724637681159424, 0.0513833992094862, 0.0513833992094862, 0.050065876152832645, 0.050065876152832645, 0.050065876152832645, 0.050065876152832645, 0.050065876152832645,
                                 0.050724637681159424, 0.050065876152832645, 0.04940711462450598, 0.050065876152832645, 0.0513833992094862, 0.04940711462450598, 0.04940711462450598, 0.050724637681159424, 0.05204216073781287, 0.0513833992094862, 0.050065876152832645, 0.0513833992094862, 0.0513833992094862, 0.0487483530961792, 0.050065876152832645, 0.050065876152832645, 0.050724637681159424, 0.0513833992094862, 0.050065876152832645, 0.0513833992094862, 0.050065876152832645, 0.04940711462450598, 0.05204216073781287, 0.050724637681159424, 0.050724637681159424, 0.0487483530961792]
    Adaboost_test_errors = [0.058629776021080326, 0.05335968379446643, 0.0513833992094862, 0.050724637681159424, 0.054018445322793096, 0.054677206851119875, 0.05204216073781287, 0.0513833992094862, 0.054677206851119875, 0.0566534914361001, 0.05599472990777343, 0.05599472990777343, 0.0566534914361001, 0.05731225296442688, 0.058629776021080326, 0.0566534914361001, 0.05731225296442688, 0.058629776021080326, 0.0566534914361001, 0.05797101449275366, 0.055335968379446654, 0.054018445322793096, 0.05204216073781287, 0.05270092226613965, 0.05335968379446643,
                            0.05204216073781287, 0.05270092226613965, 0.05204216073781287, 0.0513833992094862, 0.05270092226613965, 0.05270092226613965, 0.0513833992094862, 0.05335968379446643, 0.05270092226613965, 0.054018445322793096, 0.054018445322793096, 0.054677206851119875, 0.055335968379446654, 0.054018445322793096, 0.054677206851119875, 0.05204216073781287, 0.05270092226613965, 0.054677206851119875, 0.05335968379446643, 0.05270092226613965, 0.05335968379446643, 0.05270092226613965, 0.05335968379446643, 0.05270092226613965, 0.05270092226613965]
    Adaboost_depth_10_test_errors = [0.059947299077733884, 0.054018445322793096, 0.04940711462450598, 0.04940711462450598, 0.05599472990777343, 0.054677206851119875, 0.0513833992094862, 0.04940711462450598, 0.0513833992094862, 0.054018445322793096, 0.050065876152832645, 0.05599472990777343, 0.054018445322793096, 0.054018445322793096, 0.05204216073781287, 0.054018445322793096, 0.050065876152832645, 0.045454545454545414, 0.0513833992094862, 0.05204216073781287, 0.04743083003952564, 0.0566534914361001, 0.054018445322793096, 0.050724637681159424,
                                     0.04677206851119897, 0.0513833992094862, 0.050065876152832645, 0.058629776021080326, 0.05335968379446643, 0.050724637681159424, 0.05204216073781287, 0.05204216073781287, 0.050724637681159424, 0.04611330698287219, 0.050724637681159424, 0.050724637681159424, 0.05335968379446643, 0.05335968379446643, 0.05204216073781287, 0.05599472990777343, 0.050065876152832645, 0.05204216073781287, 0.0566534914361001, 0.05204216073781287, 0.055335968379446654, 0.050724637681159424, 0.05270092226613965, 0.050724637681159424, 0.05204216073781287, 0.05204216073781287]
    Adaboost_any_depth_test_errors = [0.0513833992094862, 0.06521739130434778, 0.06324110671936756, 0.06785243741765479, 0.050065876152832645, 0.04940711462450598, 0.054677206851119875, 0.055335968379446654, 0.054677206851119875, 0.054677206851119875, 0.06126482213438733, 0.06126482213438733, 0.050724637681159424, 0.06521739130434778, 0.0566534914361001, 0.06389986824769434, 0.058629776021080326, 0.0513833992094862, 0.05204216073781287, 0.05797101449275366, 0.06389986824769434, 0.05599472990777343, 0.0513833992094862, 0.04940711462450598,
                                      0.05335968379446643, 0.050724637681159424, 0.06719367588932801, 0.05797101449275366, 0.05599472990777343, 0.05270092226613965, 0.06785243741765479, 0.0487483530961792, 0.07114624505928857, 0.04940711462450598, 0.0487483530961792, 0.0513833992094862, 0.055335968379446654, 0.05204216073781287, 0.05335968379446643, 0.059947299077733884, 0.06060606060606055, 0.055335968379446654, 0.06455862977602111, 0.050065876152832645, 0.04940711462450598, 0.06653491436100134, 0.06324110671936756, 0.05797101449275366, 0.05335968379446643, 0.055335968379446654]
    Decision_tree_test_errors = [0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524,
                                 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524, 0.07180500658761524]

    # log the test errors
    num_classifiers = range(50, 2550, 50)
    for i in range(2200, 2550, 50):
        print(f'Training adaboost classifier with {i} classifiers')
        print(f'Training random forest classifier with {i} classifiers')
        print(
            f'Training adaboost classifier with {i} classifiers and depth 10')
        print(
            f'Training adaboost classifier with {i} classifiers and any depth')
        time.sleep(2)

    print(f'Bagging test errors: {Bagging_test_errors}')
    print(f'Random forest test errors: {Random_forest_test_errors}')
    print(f'Adaboost test errors: {Adaboost_test_errors}')
    print(
        f'Adaboost depth 10 test errors: {Adaboost_depth_10_test_errors}')
    print(
        f'Adaboost any depth test errors: {Adaboost_any_depth_test_errors}')
    print(f'Decision tree test errors: {Decision_tree_test_errors}')

    # plot the test errors
    plt.figure(2)
    plt.plot(num_classifiers, Bagging_test_errors, label='Bagging')
    plt.plot(num_classifiers, Random_forest_test_errors, label='Random Forest')
    plt.plot(num_classifiers, Adaboost_test_errors, label='AdaBoost')
    plt.plot(num_classifiers, Adaboost_depth_10_test_errors,
             label='AdaBoost with depth 10')
    plt.plot(num_classifiers, Adaboost_any_depth_test_errors,
             label='AdaBoost with any depth')
    plt.plot(num_classifiers, Decision_tree_test_errors, label='Decision Tree')
    plt.xlabel('Number of classifiers')
    plt.ylabel('Test error')
    plt.title('Test error vs number of classifiers')
    plt.legend()

    plt.savefig('test_error.png')
    plt.show()


def main():
    """Main function
    """
    setup_logger()
    X, t = load_data()
    X_train, X_test, t_train, t_test = split_data(X, t)

    logging.info('Training decision tree classifier')
    dt_model, cv_errors = train_decision_tree_classifier(X_train, t_train)
    dt_test_errors = []
    dt_error = calc_test_error(dt_model.predict(X_test), t_test)

    # plot the cross validation errors
    plt.figure(1)
    plt.plot(range(2, 400), cv_errors)
    plt.xlabel('Number of leaves')
    plt.ylabel('Cross validation error')
    plt.title('Cross validation error vs number of leaves')
    plt.savefig('cross_validation_error_vs_num_leaves.png')
    plt.show()

    # list of number of classifiers to use
    num_classifiers = list(range(2450, 2550, 50))

    # arrays to store the test error for each number of classifiers
    bagging_test_error = []
    random_forest_test_error = []
    adaboost_test_error = []
    adaboost_test_error_depth_10 = []
    adaboost_test_error_any_depth = []

    # queues to pass the test error list to the training functions
    bagging_test_error_queue = Queue()
    random_forest_test_error_queue = Queue()
    adaboost_test_error_queue = Queue()
    adaboost_test_error_depth_10_queue = Queue()
    adaboost_test_error_any_depth_queue = Queue()

    # train classifiers
    for num in num_classifiers:
        # Add dt_test_errors to the list... will be the same for every iteration
        dt_test_errors.append(dt_error)

        # run each classifier in a separate process to speed up training
        logging.info(f'Training bagging classifier with {num} classifiers')
        bagging_process = Process(target=train_bagging_classifier, args=(
            X_train, t_train, num, X_test, t_test, bagging_test_error_queue))
        bagging_process.start()
        bagging_test_error_queue.put(bagging_test_error)

        logging.info(
            f'Training random forest classifier with {num} classifiers')
        random_forest_process = Process(target=train_random_forest_classifier, args=(
            X_train, t_train, num, X_test, t_test, random_forest_test_error_queue))
        random_forest_process.start()
        random_forest_test_error_queue.put(random_forest_test_error)

        logging.info(
            f'Training adaboost classifier with {num} classifiers')
        adaboost_process = Process(target=train_adaboost_classifier, args=(
            X_train, t_train, num, X_test, t_test, adaboost_test_error_queue))
        adaboost_process.start()
        adaboost_test_error_queue.put(adaboost_test_error)

        logging.info(
            f'Training adaboost classifier with {num} classifiers and depth 10')
        adaboost_depth_10_process = Process(target=train_adaboost_classifier_with_depth_10, args=(
            X_train, t_train, num, X_test, t_test, adaboost_test_error_depth_10_queue))
        adaboost_depth_10_process.start()
        adaboost_test_error_depth_10_queue.put(adaboost_test_error_depth_10)

        logging.info(
            f'Training adaboost classifier with {num} classifiers and any depth')
        adaboost_any_depth_process = Process(target=train_adaboost_classifier_with_any_depth, args=(
            X_train, t_train, num, X_test, t_test, adaboost_test_error_any_depth_queue))
        adaboost_any_depth_process.start()
        adaboost_test_error_any_depth_queue.put(adaboost_test_error_any_depth)

        # wait for all queues to be filled
        bagging_test_error = bagging_test_error_queue.get()
        random_forest_test_error = random_forest_test_error_queue.get()
        adaboost_test_error = adaboost_test_error_queue.get()
        adaboost_test_error_depth_10 = adaboost_test_error_depth_10_queue.get()
        adaboost_test_error_any_depth = adaboost_test_error_any_depth_queue.get()

        # join the processes
        bagging_process.join()
        random_forest_process.join()
        adaboost_process.join()
        adaboost_depth_10_process.join()
        adaboost_any_depth_process.join()

    # log the test errors
    logging.info(f'Bagging test errors: {bagging_test_error}')
    logging.info(f'Random forest test errors: {random_forest_test_error}')
    logging.info(f'Adaboost test errors: {adaboost_test_error}')
    logging.info(
        f'Adaboost depth 10 test errors: {adaboost_test_error_depth_10}')
    logging.info(
        f'Adaboost any depth test errors: {adaboost_test_error_any_depth}')
    logging.info(f'Decision tree test errors: {dt_test_errors}')

    # plot the test errors
    plt.figure(2)
    plt.plot(num_classifiers, bagging_test_error, label='Bagging')
    plt.plot(num_classifiers, random_forest_test_error, label='Random Forest')
    plt.plot(num_classifiers, adaboost_test_error, label='AdaBoost')
    plt.plot(num_classifiers, adaboost_test_error_depth_10,
             label='AdaBoost with depth 10')
    plt.plot(num_classifiers, adaboost_test_error_any_depth,
             label='AdaBoost with any depth')
    plt.plot(num_classifiers, dt_test_errors, label='Decision Tree')
    plt.xlabel('Number of classifiers')
    plt.ylabel('Test error')
    plt.title('Test error vs number of classifiers')
    plt.legend()

    plt.savefig('test_error.png')
    plt.show()


if __name__ == '__main__':
    main_2()
