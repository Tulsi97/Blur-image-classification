import cv2
import numpy as np
import pandas as pd
import os
import pickle
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score


input_size = (512, 512)
X_test = []
y_test = []
y_pred = []
t_blur = []
t_nblur = []
threshold_val = 450

#  compute the Laplacian of the image
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


digital_blur_set = pd.read_excel('CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet.xlsx')
natural_blur_set = pd.read_excel('CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet.xlsx')


digital_blur_set['MyDigital Blur'] = digital_blur_set['MyDigital Blur'].apply(lambda x : x.strip())
digital_blur_set = digital_blur_set.rename(index=str, columns={"Unnamed: 1": "Blur Label"})
natural_blur_set['Image Name'] = natural_blur_set['Image Name'].apply(lambda x : x.strip())



folder_path = 'CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet/'

import numpy as np

from scipy.ndimage import variance
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import laplace
from skimage.transform import resize

Var = []
Max_val = []
Variance_laplacian = []

# load image arrays
for filename in os.listdir(folder_path):
    if filename != '.DS_Store':
        image_path = folder_path + filename
        img = image.load_img(image_path, target_size= input_size)

        blur = digital_blur_set[digital_blur_set['MyDigital Blur'] == filename].iloc[0]['Blur Label']
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)

        #Grayscaleimage
        #img = rgb2gray(img)

        # Edge detection
        edge_laplace = laplace(gray, ksize=3)

        # Print
        #print(f"Variance: {variance(edge_laplace)}")
        #print(f"Maximum : {np.amax(edge_laplace)}")

        fm = variance_of_laplacian(gray)

        #print("fm:", fm)
        Var.append(variance(edge_laplace))
        Max_val.append(np.amax(edge_laplace))
        Variance_laplacian.append(fm)


        if fm < threshold_val:
            y_pred.append(1)
        else:
            y_pred.append(0)

        if blur == 1:
            y_test.append(1)
            t_blur.append(fm)
        else:
            y_test.append(0)
            t_nblur.append(fm)
    else:
        print(filename, 'not a pic')

print("Artificially Blurred Evaluated...")


print("Var max:", max(Var))
print("Max_val:", max(Max_val))
print("Var_lap max:", max(Variance_laplacian))

folder_path = 'CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet/'

# load image arrays
for filename in os.listdir(folder_path):
    if filename != '.DS_Store':
        image_path = folder_path + filename
        img = image.load_img(image_path, target_size=input_size)

        blur = natural_blur_set[natural_blur_set['Image Name'] == filename.split('.')[0]].iloc[0]['Blur Label']
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        if fm < threshold_val:
            y_pred.append(1)
        else:
            y_pred.append(0)
        if blur == 1:
            y_test.append(1)
            t_blur.append(fm)
        else:
            y_test.append(0)
            t_nblur.append(fm)
    else:
        print(filename, 'not a pic')

print("Naturally Blurred Evaluated images ...")

#loading test pickle file
with open('y_test.pkl', 'rb') as picklefile:
    y_test = pickle.load(picklefile)


accuracy = accuracy_score(y_test, y_pred)
print("accuracy: {:.2f}%".format(accuracy * 100))