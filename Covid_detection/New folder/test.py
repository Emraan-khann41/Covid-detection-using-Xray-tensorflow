# USAGE
# python train.py --dataset dataset

# import the necessary packages
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

INIT_LR = 1e-3
EPOCHS = 20
BS = 8

# construct the argument parser and parse the arguments


print("[INFO] loading images...")
imagePaths = list(paths.list_images('E:/Thesis___/practise_vgg/big_data2'))
data = []
labels = []
# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]
	# load the image, swap color channels, and resize it to be a fixed
	# 224x224 pixels while ignoring aspect ratio
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))
	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)
# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 1]
data = np.array(data) / 255.0
labels = np.array(labels)


# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)
# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=15,
	fill_mode="nearest")




model = tf.keras.models.load_model('model_new.h5')

predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

#cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
#total = sum(sum(cm))
#acc = (cm[0, 0] + cm[1, 1]) / total
#sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
#specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
# show the confusion matrix, accuracy, sensitivity, and specificity
"""
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))
"""


from mlxtend.plotting import plot_confusion_matrix
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
plot_confusion_matrix(conf_mat=cm)
total = sum(sum(cm))
acc = ((cm[0, 0] + cm[1, 1]) / total)*100
#acc = accuracy_score(testY, np.round(predIdxs))*100
#cm = confusion_matrix(testY, np.round(predIdxs))
tn, fp, fn, tp = cm.ravel()


print('\nCONFUSION MATRIX FORMAT ------------------\n')
print("[true positives    false positives]")
print("[false negatives    true negatives]\n\n")

print('CONFUSION MATRIX ------------------')
print(cm)



print('\nTEST METRICS ----------------------')
precision = tp/(tp+fp)*100
recall = tp/(tp+fn)*100
specificity = tn/(tn+fp)*100
print('Accuracy: {}%'.format(acc))
print('Precision: {}%'.format(precision))
print('Recall/Sensitivity: {}%'.format(recall))
print('Specificity {}%'.format(specificity))
print('F1-score: {}'.format(2*precision*recall/(precision+recall)))
