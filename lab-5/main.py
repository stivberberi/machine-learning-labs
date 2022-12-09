import numpy as np

'''
Neural network classifier from scratch with 2 hidden layers and n_1 n_2 neurons in each layer.
Uses ReLU activation function for hidden layers. Stochastic gradient descent is used with 
a learning rate of 0.005.
'''


def neural_network_classifier(X_train, y_train, X_test, y_test, n_1, n_2, epochs):
    # Initialize weights and biases
    W1 = np.random.randn(n_1, X_train.shape[0]) * 0.01
    b1 = np.zeros((n_1, 1))
    W2 = np.random.randn(n_2, n_1) * 0.01
    b2 = np.zeros((n_2, 1))
    W3 = np.random.randn(1, n_2) * 0.01
    b3 = np.zeros((1, 1))

    # Initialize lists to store cost and accuracy
    costs = []
    train_acc = []
    test_acc = []

    for i in range(epochs):
        # Forward propagation
        Z1 = np.dot(W1, X_train) + b1
        A1 = np.maximum(0, Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = np.maximum(0, Z2)
        Z3 = np.dot(W3, A2) + b3
        A3 = sigmoid(Z3)

        # Compute cost
        cost = compute_cost(A3, y_train)

        # Backward propagation
        dZ3 = A3 - y_train
        dW3 = (1 / X_train.shape[1]) * np.dot(dZ3, A2.T)
        db3 = (1 / X_train.shape[1]) * np.sum(dZ3, axis=1, keepdims=True)
        dA2 = np.dot(W3.T, dZ3)
        dZ2 = np.multiply(dA2, np.int64(A2 > 0))
        dW2 = (1 / X_train.shape[1]) * np.dot(dZ2, A1.T)
        db2 = (1 / X_train.shape[1]) * np.sum(dZ2, axis=1, keepdims=True)
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        dW1 = (1 / X_train.shape[1]) * np.dot(dZ1, X_train.T)
        db1 = (1 / X_train.shape[1])
