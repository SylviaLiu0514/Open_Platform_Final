import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model, Sequential
from keras.optimizers import SGD, RMSprop, Adam
from PIL import Image


def Read_train_data():
    train_data = pd.read_csv('train.csv')
    ID = train_data.pop('ID')
    species = train_data.pop('Class')
    return ID, species
                              
ID, species = Read_train_data()

print(ID)
print(species)
print(np.around(output_image / 255.0))