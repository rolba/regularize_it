"""
This is an experiment with different dropout layers placeholding in model.
I will focus only on two:
model.add(Dropout(0.5))
model.add(SpatialDropout2D(0.5))

No other regularyzation technique is used! It is not the point to get high F1-score.
The idea is to find a right place in my model for those two layers. I will not fight for high accuracy.

You will find more at:
https://ai-experiments.com/use-your-spatial-dropout-layers-wisely/‎

"""

import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, SpatialDropout2D, Dropout
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

labelsNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

(xTrain, yTrain), (xTest, yTest) = tensorflow.keras.datasets.cifar10.load_data()      # Load data
xTrain, xVal, yTrain, yVal = train_test_split(xTrain, yTrain, test_size = 0.15)       # Split validation set

input_shape = (32, 32, 3)

xTrain = xTrain.reshape(xTrain.shape[0], 32, 32, 3)
xTest = xTest.reshape(xTest.shape[0], 32, 32, 3)
xVal = xVal.reshape(xVal.shape[0], 32, 32, 3)

# Normalize data adn treat them as floats.
xTrain = xTrain.astype('float32') / 255
xTest = xTest.astype('float32') / 255
xVal = xVal.astype('float32') / 255

#Encode labels
lb = LabelBinarizer()
yTrain = lb.fit_transform(yTrain)
yTest = lb.fit_transform(yTest)
yVal = lb.fit_transform(yVal)

# You can notice that I commented dropout layers placeholders. And I will put there different dropout layers,
# train mode, read chart and again and again. Changin only dropout layer placement.
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation=tensorflow.nn.relu, input_shape=input_shape))
# "Dropout placeholder 1"
model.add(Conv2D(64, kernel_size=(3, 3), activation=tensorflow.nn.relu))
# "Dropout placeholder 2"
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation=tensorflow.nn.relu))
# "Dropout placeholder 3"
model.add(Conv2D(128, kernel_size=(3, 3), activation=tensorflow.nn.relu))
# "Dropout placeholder 4"
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=tensorflow.nn.relu))
# "Dropout placeholder 5"
model.add(Dense(64, activation=tensorflow.nn.relu))
# "Dropout placeholder 6"
model.add(Dense(10, activation=tensorflow.nn.softmax))

# To speed it up I will train only for 20 epochs.
# I am aware that it is not enough. But I have only a GTX1060,
# no automatic framework for this experiment and finally all is done manually.
epochs = 20

# I will use Adam with 0.001 learning rate. It's a good idea for experimenting with regularization placement.
opt = tensorflow.keras.optimizers.Adam(0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(xTrain,
         yTrain,
         batch_size=256,
         epochs=epochs,
         validation_data=(xVal, yVal))

# Print training and validation loss and accuracy
plt.style.use("ggplot")
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

yPredictions = model.predict(xTest, batch_size=256)
report = classification_report(yTest.argmax(axis=1), yPredictions.argmax(axis=1), target_names=labelsNames)
print(report)

yPred = lb.inverse_transform(yPredictions)
yTrue = lb.inverse_transform(yTest)
cm = confusion_matrix(yTrue, yPred)

fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(cm, cmap=plt.cm.Blues)

ax.xaxis.set_ticklabels(labelsNames);
ax.yaxis.set_ticklabels(labelsNames);

for i in range(10):
    for j in range(10):
        c = cm[j,i]
        ax.text(i, j, str(c), va='center', ha='center')


# Print confusion matrix
plt.xticks(range(10))
plt.yticks(range(10))
plt.suptitle('Confusion matrix',size = 32)
plt.xlabel('True labeling',size = 32)
plt.ylabel('Predicted labeling',size = 32)
plt.rcParams.update({'font.size': 28})
plt.show()
