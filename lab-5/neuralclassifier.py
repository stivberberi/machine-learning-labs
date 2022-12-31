import numpy as np


def cross_entropy_loss(y, y_pred):
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


class NeuralClassifier:
    def __init__(self, n1, n2, X_train, learning_rate=0.005):
        # Initialize the network weights and biases with random values
        self.input_size = X_train.shape[1]
        self.output_size = 1
        self.hidden_size1 = n1
        self.hidden_size2 = n2
        self.learning_rate = learning_rate

        # Weights
        self.w1 = np.random.randn(self.input_size, self.hidden_size1)
        self.w2 = np.random.randn(self.hidden_size1, self.hidden_size2)
        self.w3 = np.random.randn(self.hidden_size2, self.output_size)

        # Save the best weights and biases
        self.best_w1 = self.w1
        self.best_w2 = self.w2
        self.best_w3 = self.w3

        # Initialize arrays to store test and validation cross entropy losses
        self.test_loss = []
        self.val_loss = []
        self.best_val_loss = 1000

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Sigmoid derivative
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # ReLU activation function
    def relu(self, x):
        return np.maximum(0, x)

    # ReLU derivative
    def relu_derivative(self, x):
        return (x > 0).astype(int)

    # Forward propagation
    def forward(self, X):
        self.z1 = np.dot(X, self.w1)
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2)
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.w3)
        output = self.sigmoid(self.z3)
        return output

    # Backpropagation
    def backward(self, X, y, output):
        error = y - output
        delta3 = error * self.sigmoid_derivative(output)
        dJdw3 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.w3.T) * self.relu_derivative(self.a2)
        dJdw2 = np.dot(self.a1.T, delta2)

        delta1 = np.dot(delta2, self.w2.T) * self.relu_derivative(self.a1)

        regularization_term = 0.01 * np.sum(self.w1)
        dJdw1 = np.dot(X.T, delta1)

        self.w1 += self.learning_rate * (dJdw1 + regularization_term)
        self.w2 += self.learning_rate * (dJdw2 + regularization_term)
        self.w3 += self.learning_rate * (dJdw3 + regularization_term)

    def train(self, X, y, X_val, y_val, num_epochs=100):
        # Train the network using stochastic gradient descent
        for _ in range(num_epochs):
            # Shuffle the training data
            X = X[np.random.permutation(X.shape[0])]

            # Train the network
            output = self.forward(X)
            self.backward(X, y, output)

            # Compute the losses
            y_pred = self.forward(X)
            self.test_loss.append(cross_entropy_loss(y, y_pred))
            self.val_loss.append(cross_entropy_loss(
                y_val, self.forward(X_val)))

            # Save the best weights
            if self.val_loss[-1] < self.best_val_loss:
                self.best_val_loss = self.val_loss[-1]
                self.best_w1 = self.w1
                self.best_w2 = self.w2
                self.best_w3 = self.w3

    def get_validation_loss(self):
        return self.best_val_loss

    def predict(self, X):
        # Perform a forward pass with the best weights
        z1 = np.dot(X, self.best_w1)
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.best_w2)
        a2 = self.relu(z2)
        z3 = np.dot(a2, self.best_w3)
        output = self.sigmoid(z3)

        return output

    def get_best_weights(self):
        return self.best_w1, self.best_w2, self.best_w3

    def get_test_loss(self):
        return self.test_loss
