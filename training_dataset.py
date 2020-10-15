import numpy as np
import pandas as pd
import os
import pickle
from keras.preprocessing import image

X_train = []
y_train = []

input_size = (128, 128)

# Training set file path
folder_path = 'CERTH_ImageBlurDataset/TrainingSet/Undistorted/'


# load image arrays of undistorted image
for filename in os.listdir(folder_path):
    if filename != '.DS_Store':
        image_path = folder_path + filename
        img = image.load_img(image_path, target_size = input_size)
        X_train.append((1/255)*np.asarray(img))
        y_train.append(0)
    else:
        print(filename, 'not a pic')
print("Training set : Undistorted images loaded...")


# training set of Artificially blurred image
folder_path = 'CERTH_ImageBlurDataset/TrainingSet/Artificially-Blurred/'

# load image arrays Artificially blurred image
for filename in os.listdir(folder_path):
    if filename != '.DS_Store':
        image_path = folder_path + filename
        img = image.load_img(image_path, target_size=input_size)
        X_train.append((1/255)*np.asarray(img))
        y_train.append(1)
    else:
        print(filename, 'not a pic')
print("Training set: Artificially Blurred images loaded...")


# training set of Naturally Blurred images
folder_path = 'CERTH_ImageBlurDataset/TrainingSet/Naturally-Blurred/'

# load image arrays Naturally Blurred images
for filename in os.listdir(folder_path):
    if filename != '.DS_Store':
        image_path = folder_path + filename
        img = image.load_img(image_path, target_size=input_size)
        X_train.append((1/255)*np.asarray(img))
        y_train.append(1)
    else:
        print(filename, 'not a pic')
print("Training set: Naturally Blurred images loaded...")


# Pickle the train files
with open('X_train.pkl', 'wb') as picklefile:
    pickle.dump(X_train, picklefile)


with open('y_train.pkl', 'wb') as picklefile:
    pickle.dump(y_train, picklefile)