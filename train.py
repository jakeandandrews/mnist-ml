import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from nn import NeuralNetwork
import random


# -------- DATA PREP ---------------------
class DataInitialiser: 
    def __init__ (self):
        self.df = pd.read_csv('mnist_examples.csv')

        data = np.array(self.df)
        self.m, self.n = data.shape #n is number of features +1 cuz of labels
        np.random.shuffle(data)

        data_validation = data[0:1000].T
        self.y_val = data_validation[0] # first row is labels
        self.X_val = data_validation[1:self.n]
        self.X_dev = self.X_val / 255.

        data_train= data[1000:self.m].T
        self.y_train = data_train[0] # first row is labels
        self.X_train = data_train[1:self.n] 
        self.X_train = self.X_train / 255.
        _,self.m_train = self.X_train.shape

        # y_train, len(y_train) # array of all our labels for the training data
        # X_train[:, 0].shape  # first column has 784 pixels in it
        #----------IMPLEMENT NN ------------------

# ---------- TEST ---------------------------
# rand_num = random.randint(0, m)
# nn.test_prediction(rand_num, X_train, y_train, plot_it=True, print_out=True)


