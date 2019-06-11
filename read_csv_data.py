import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import load_image



def Read_train_data():
    # Read train.csv and pop out the id
    train_data = pd.read_csv('train.csv')
    ID = train_data.pop('ID')
    # Pop out the species and make species name correspond to number 
    species = train_data.pop('Class')
    species = LabelEncoder().fit(species).transform(species)
    # Standardize the data by setting the mean to 0 and std to 1
    return ID, species
                              
def load_train_data():
    # Load the train data
    ID, label = Read_train_data()
    # Load the image data
    train_image = load_image.Load_image(ID)
    # Split them into validation and cross-validation
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2)
    train_id, test_id = next(sss.split(train_image, label))
    tra_image, tra_label = train_image[train_id], label[train_id]
    val_image, val_label = train_image[test_id], label[test_id]
    return (tra_image, tra_label), (val_image, val_label)

<<<<<<< HEAD
#print(load_train_data())
(tra_image, tra_label), (val_image, val_label) = load_train_data()
=======
print(ID)
print(species)
>>>>>>> df70b88b079d9a8fcf04189b1d135eece95eb865
