import numpy as np


def relu(x):
    return np.maximum(0, x)


def cross_entropy_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred))


class NeuralNetwork:
    def __init__(self, n1, n2):
        # Initialize the network weights and biases with random values
        self.w1 = np.random.randn(n1, X_train.shape[1])
        self.b1 = np.zeros(n1)
        self.w2 = np.random.randn(n2, n1)
        self.b2 = np.zeros(n2)
        self.w3 = np.random.randn(1, n2)
        self.b3 = 0

    def forward(self, x):
        # Perform a forward pass through the network
        z1 = x.dot(self.w1.T) + self.b1
        a1 = relu(z1)
        z2 = a1.dot(self.w2.T) + self.b2
        a2 = relu(z2)
        z3 = a2.dot(self.w3.T) + self.b3
        a3 = sigmoid(z3)
        return a3

    def train(self, X, y, learning_rate=0.005, num_epochs=100):
        # Train the network using stochastic gradient descent
        for epoch in range(num_epochs):
            # Perform a forward pass through the network
            y_pred = self.forward(X)

            # Compute the loss
            loss = cross_entropy_loss(y, y_pred)

            # Print the current loss
            print(f"Epoch {epoch+1}: Loss = {loss}")

            # Perform a backward pass through the network
            delta3 = y_pred - y
            dw3 = (a2.T).dot(delta3)
            db3 = np.sum(delta3)
            delta2 = delta3.dot(self.w3) * (a2 > 0)
            dw2 = (a1.T).dot(delta2)
            db2 = np.sum(delta2)
            delta1 = delta2.dot(self.w2) * (a1 > 0)
            dw1 = (x.T).dot(delta1)
            db1 = np.sum(delta1)

            # Update the network weights and biases
            self.w1 -= learning_rate * dw1
            self.b1 -= learning_rate * db1
            self.w2 -= learning_rate * dw2
            self.b2 -= learning_rate * db2
            self.w3 -= learning_rate * dw3
            self.b3 -= learning_rate * db3
