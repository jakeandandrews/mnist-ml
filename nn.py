import numpy as np
from matplotlib import pyplot as plt
import random

class NeuralNetwork: 
    def __init__(self, alpha=0.1):
        self.W1, self.b1, self.W2, self.b2 = self.init_params()
        self.alpha = alpha
        self.accuracy = 0.0 
        self.iteration = 0

    def init_params(self):
        W1 = np.random.rand(10, 784) - 0.5 #this will be between -0.5 and 0.5
        b1 = np.random.rand(10, 1) - 0.5
        W2 = np.random.rand(10,10) - 0.5
        b2 = np.random.rand(10,1) - 0.5
        return W1, b1, W2, b2

    def train(self, X, Y, iterations, print_out=True):
        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_prop(X)
            dW1, db1, dW2, db2 = self.back_prop(Z1, A1, Z2, A2, X, Y)
            self.update_params(dW1, db1, dW2, db2)
            self.accuracy = self.get_accuracy(self.get_predictions(A2), Y)
            self.iteration += 1
            if print_out:
                if self.iteration % 10 == 0: #every tenth iteration we will print
                    print("Iteration: ", self.iteration)
                    print("Accuracy: ", self.accuracy)

    def forward_prop(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.ReLU(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def back_prop(self, Z1, A1, Z2, A2, X, Y):
        m = Y.size
        one_hot_Y = self.one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = self.W2.T.dot(dZ2) * self.deriv_ReLU(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2):
        self.W1 -= self.alpha * dW1
        self.b1 -= self.alpha * db1
        self.W2 -= self.alpha * dW2
        self.b2 -= self.alpha * db2

    def ReLU(self, Z):
        return np.maximum(0, Z)

    def deriv_ReLU(self, Z):
        return Z > 0 #works cuz of bools

    def softmax(self, Z):
        A = np.exp(Z)/ sum(np.exp(Z))
        return A

    def one_hot(self, Y):

        one_hot_Y = np.zeros((Y.size, Y.max()+1)) #m (number of examples) Y.max is 9 + 1=> 10. so this creates correct matrix size
        one_hot_Y[np.arange(Y.size),Y] = 1 #for each row, go to the row specified by Y and set it to Y
        one_hot_Y= one_hot_Y.T #we transpose it because we want columns to be examples
        return one_hot_Y

    def get_predictions(self, A2):
        return np.argmax(A2, axis=0)

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size
    
    def make_predictions(self, X):
        _, _, _, A2 = self.forward_prop(X)
        predictions = self.get_predictions(A2)
        return predictions

    def test_prediction(self, index, X_train, y_train, plot_it=True, print_out=True):
        current_image = X_train[:, index, None]
        prediction = self.make_predictions(X_train[:, index, None])
        label = y_train[index]
        if print_out: 
            print("Prediction: ", prediction)
            print("Label: ", label)
        if plot_it:
            current_image = current_image.reshape((28, 28)) * 255
            plt.gray()
            plt.title(f"prediction: {prediction[0]} Actual: {label}")
            plt.imshow(current_image, interpolation='nearest')
            plt.show()
        return prediction, label   

    def guess_my_digit(self, X):
        _, _, _, likelihoods = self.forward_prop(X)
        predictions = self.get_predictions(likelihoods)
        return predictions, likelihoods