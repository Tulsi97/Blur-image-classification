import numpy as np
import pandas as pd
import os
import pickle
from keras.preprocessing import image


input_size = (128, 128)

X_test = []
y_test = []

digital_blur_set = pd.read_excel('CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet.xlsx')
natural_blur_set = pd.read_excel('CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet.xlsx')



digital_blur_set['MyDigital Blur'] = digital_blur_set['MyDigital Blur'].apply(lambda x: x.strip())

digital_blur_set = digital_blur_set.rename(index=str, columns={"Unnamed: 1": "Blur Label"})


natural_blur_set['Image Name'] = natural_blur_set['Image Name'].apply(lambda x: x.strip())




folder_path = 'CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet/'

# load image arrays
for filename in os.listdir(folder_path):
    if filename != '.DS_Store':
        image_path = folder_path + filename
        img = image.load_img(image_path, target_size=input_size)
        X_test.append((1/255)*np.asarray(img))
        blur = digital_blur_set[digital_blur_set['MyDigital Blur'] == filename].iloc[0]['Blur Label']
        if blur == 1:
            y_test.append(1)
        else:
            y_test.append(0)
    else:
        print(filename, 'not a pic')
print("Test set: Artificially Blurred images loaded...")


folder_path = 'CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet/'

# load image arrays
for filename in os.listdir(folder_path):
    if filename != '.DS_Store':
        image_path = folder_path + filename
        img = image.load_img(image_path, target_size=input_size)
        X_test.append((1/255)*np.asarray(img))
        blur = natural_blur_set[natural_blur_set['Image Name'] == filename.split('.')[0]].iloc[0]['Blur Label']
        if blur == 1:
            y_test.append(1)
        else:
            y_test.append(0)
    else:
        print(filename, 'not a pic')

print("Training set: Naturally Blurred images loaded...")



with open('X_test.pkl', 'wb') as picklefile:
    pickle.dump(X_test, picklefile)


with open('y_test.pkl', 'wb') as picklefile:
    pickle.dump(y_test, picklefile)