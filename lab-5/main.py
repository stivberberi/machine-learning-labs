import numpy as np
import matplotlib.pyplot as plt
from neuralclassifier import NeuralClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import logging


def setup_logger():
    '''
    Set up the logger
    '''
    # setup logger for writing to file
    logging.basicConfig(filename='lab_5.log', filemode='w',
                        level=logging.INFO, format='%(message)s')
    # add a stream handler to also send the output to stdout
    logging.getLogger().addHandler(logging.StreamHandler())


def load_split_data():
    '''
    Load the data from the data_banknote_authentication.txt file
    Split the data 60/20/20 into training, validation, and test sets
    '''
    # Load the data
    data = np.loadtxt("data_banknote_authentication.txt", delimiter=",")
    # shuffle the data wtih a fixed seed
    data = data[np.random.RandomState(seed=2350).permutation(data.shape[0])]

    X = data[:, :-1]
    y = data[:, -1]

    # Make y a column vector
    y = y.reshape(-1, 1)

    # Standardize the data
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # split the data into training, validation, and test sets
    X_train = X[:900]
    y_train = y[:900]
    X_val = X[900:1100]
    y_val = y[900:1100]
    X_test = X[1100:]
    y_test = y[1100:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(n1, n2, X_train, y_train, X_val, y_val):
    '''
    Train a neural network with n1 neurons in the first layer, n2 neurons in the
    second layer, and 100 epochs. Trains the network 5 times and returns the network with the
    lowest validation loss.
    '''
    # Train the network 5 times and return the network with the lowest validation loss
    for i in range(5):
        # Create a new neural network
        nn = NeuralClassifier(n1, n2, X_train)

        # Train the neural network
        nn.train(X_train, y_train, X_val, y_val)

        # Find the network with the lowest validation loss
        if i == 0:
            best_nn = nn
        elif nn.get_validation_loss() < best_nn.get_validation_loss():
            best_nn = nn

    return best_nn


def main():
    setup_logger()

    X_train, y_train, X_val, y_val, X_test, y_test = load_split_data()

    # Test n1 and n2 values between 1 and 10
    n1_values = range(1, 10)
    n2_values = range(1, 10)
    lowest_val_loss = 1000
    best_n1 = 0
    best_n2 = 0
    best_misclassification_rate = 0
    for n1 in n1_values:
        for n2 in n2_values:
            # Train the model
            nn = train_model(n1, n2, X_train, y_train, X_val, y_val)

            # Calculate the misclassification rate, and round each value to 0 or 1
            y_pred = nn.predict(X_test)
            y_pred = np.round(y_pred)
            misclassification_rate = np.mean(y_pred != y_test) * 100

            # Log the results
            logging.info("n1: {}, n2: {}, validation loss: {}, misclassification rate: {}".format(
                n1, n2, nn.get_validation_loss(), misclassification_rate))

            # Save the model with the lowest validation loss
            if nn.get_validation_loss() < lowest_val_loss:
                lowest_val_loss = nn.get_validation_loss()
                best_misclassification_rate = misclassification_rate
                best_n1 = n1
                best_n2 = n2
                best_nn = nn

    # Log the best model paramaters
    w1, w2, w3 = best_nn.get_best_weights()
    logging.info(f"Best model had n1: {best_n1}, n2: {best_n2}")
    logging.info(
        f"Best model weights: w1: {w1}, w2: {w2}, w3: {w3}")
    logging.info(
        f"Best model misclassification rate: {best_misclassification_rate}")
    logging.info(f"Best model validation loss: {lowest_val_loss}")

    # Plot the loss of best model (learning curve)
    plt.plot(best_nn.get_test_loss(), label="Training Loss")
    plt.title("Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("learning_curve.png")


if __name__ == "__main__":
    main()
