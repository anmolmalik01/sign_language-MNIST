# ======================================= dependencies =======================================
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization

# training and testing data
train_df = pd.read_csv("sign_mnist_train.csv")
test_df = pd.read_csv("sign_mnist_test.csv")

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
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

print(train_df.head())

plt.imshow(x_train[10].reshape(28, 28) , cmap = "gray")

# ===================================== model ==================================== 
model = Sequential()

model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 512 , activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 24 , activation = 'softmax'))

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

history = model.fit(x_train,y_train, batch_size = 128 ,epochs = 5 , validation_data = (x_test, y_test) )


print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

classifications = model.predict(x_test)
plt.imshow(x_test[1].reshape(28, 28) , cmap = "gray")

print(classifications[1])

print(y_test[1])

# saving and loading the .h5 model
 
# save model
model.save('gfgModel.h5')
print('Model Saved!')
 

# =================================== plots ====================================
pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot()
plt.title("Accuracy", fontsize = 15)
plt.ylim(0,1.1)

plt.show()



pd.DataFrame(history.history)[['loss','val_loss']].plot()
plt.title("Loss", fontsize = 15)

plt.show()



train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs = [i for i in range(5)]

plt.plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
plt.plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.show()



epochs = [i for i in range(5)]

plt.plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
plt.plot(epochs , val_loss , 'r-o' , label = 'Testing Loss')
plt.title('Testing Accuracy & Loss')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.show()