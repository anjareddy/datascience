# ######################################################
# Code 3-4
# MLP w/ MNIST
# Machine Learning and Deep Learning Course
# Kent State University
# Jungyoon Kim, Ph.D.
# ######################################################
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np

# Read the MNIST dataset and split it into a training set and a test set.
mnist_ds=fetch_openml('mnist_784')
mnist_ds.data=mnist_ds.data/255.0
x_train=mnist_ds.data[:50000]; x_test=mnist_ds.data[50000:]
y_train=np.int16(mnist_ds.target[:50000]); y_test=np.int16(mnist_ds.target[50000:])

# Training MLP Classifier Model
mlpC=MLPClassifier(hidden_layer_sizes=(120),learning_rate_init=0.001,batch_size=512,max_iter=250,solver='adam',verbose=True)
mlpC.fit(x_train,y_train)

# Prediction with Test Set
res=mlpC.predict(x_test)

# Confusion Matrix
confM=np.zeros((10,10),dtype=np.int16)
for i in range(len(res)):
    confM[res[i]][y_test[i]]+=1
print(confM)

# Calculating Accuracy
no_correct=0
for i in range(10):
    no_correct+=confM[i][i]
accuracy=no_correct/len(res)
print("Accuracy for Test Set is ", accuracy*100, "%.")