import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model, Sequential
from keras.optimizers import SGD, RMSprop, Adam
from PIL import Image

def Load_image(ID, image_length = 512):
    output_image = np.empty((len(ID), image_length, image_length, 1))
    for i, image_ID in enumerate(ID):
        image = img_to_array(resize_image(load_img('spectrogram/' + str(image_ID) + '.png', color_mode = "grayscale"), 
                                                    image_length=image_length))

        h1 = int((image_length - image.shape[0] ) / 2)
        h2 = h1 + image.shape[0]
        w1 = int((image_length - image.shape[1]) / 2)
        w2 = w1 + image.shape[1]
        output_image[i, h1:h2, w1:w2, 0:1] = image
    return np.around(output_image / 255.0)

def Read_train_data():
    train_data = pd.read_csv('train.csv')
    ID = train_data.pop('ID')
    species = train_data.pop('Class')
    return ID, species
                              
ID, species = Read_train_data()

print(ID)
print(species)
print(np.around(output_image / 255.0))