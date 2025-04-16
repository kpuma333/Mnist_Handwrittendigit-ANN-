🔹 Imports
import numpy as np
import pandas as pd
import keras
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Flatten
Numpy, Pandas – standard for data manipulation.

Seaborn, Matplotlib – for visualization.

Sklearn – for preprocessing and data splitting.

Keras – for building the neural network.

🔹 Load the MNIST Dataset
from keras.datasets import mnist
This imports the built-in MNIST dataset from Keras.

🔹 Understanding the MNIST Dataset
len(mnist.load_data())  # Output: 2
This tells you that mnist.load_data() returns a tuple of two elements:

Training set: (x_train, y_train)

Test set: (x_test, y_test)

🔹 Preview of Data
mnist.load_data()[0]
This shows the training data: (x_train, y_train).

x_train is an array of shape (60000, 28, 28) – 60,000 grayscale images of handwritten digits (28x28 pixels).

y_train is an array of shape (60000,) – the corresponding labels (0 to 9).

