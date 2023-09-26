import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model


# training and testing data
train_df = pd.read_csv('./data/sign_mnist_train.csv')
test_df = pd.read_csv('./data/sign_mnist_test.csv')

y_train = train_df['label']
y_test = test_df['label']

# deleting label column
del train_df['label']
del test_df['label']

# transforming categrical column(label) y
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)

# features x
x_train = train_df.values
x_test = test_df.values

# Normalize the data
x_train = x_train / 255
x_test = x_test / 255
# Reshaping the data from 1-D to 3-D as required through input by CNN's
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# load model
savedModel = load_model('gfgModel.h5')
# savedModel.summary()

# a = [ 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' ]


# prediction on test set
a = x_test[0].reshape(28, 28)
cv2.imshow('image',  a)
predictions = savedModel.predict(x_test)

print(predictions[0])

# max_value = max(predictions[0])
# max_value = max_value.tolist()
# max_index = predictions[0].index(max_value)
# print(max_index)

print(y_test[0])


# # prediction on image created
# im = cv2.imread("./backup/3.jpg")
# im = im / 255
# im = im.reshape(-1,28,28,1)
# predictions = savedModel.predict(im)
# print(predictions)
