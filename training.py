import pandas as pd
import pylab
import numpy as np
import tensorflow as tf
import os
import gc
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model, Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, Input, merge
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from keras.utils.np_utils import to_categorical
from PIL import Image

def resize_image(image, image_length = 512):
    # Find deflation_rate for narrow the image
    deflation_rate = image_length / float(image.size[max((0, 1), key=lambda i: image.size[i])])
    image.resize((int(image.size[0] * deflation_rate), int(image.size[1] * deflation_rate)))
    return image.resize((int(image.size[0] * deflation_rate), int(image.size[1] * deflation_rate)))
def Load_image(ID, image_length = 512):
    output_image = np.empty((len(ID), image_length, image_length, 1))
    # i -> the i th of the ID set, image_ID -> the name of the image
    for i, image_ID in enumerate(ID):
        # Turn the image into an array
        image = img_to_array(resize_image(load_img('spectrogram/' + str(image_ID) + '.png', color_mode = "grayscale"), 
                                                    image_length=image_length))
        # Get image height and width
        # Place the image at the center of the image
        h1 = int((image_length - image.shape[0] ) / 2)
        h2 = h1 + image.shape[0]
        w1 = int((image_length - image.shape[1]) / 2)
        w2 = w1 + image.shape[1]
        # Insert into image matrix
        output_image[i, h1:h2, w1:w2, 0:1] = image
    # Scale the array values so they are between 0 and 1
    return np.around(output_image / 255.0)

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
    train_image = Load_image(ID)
    # Split them into validation and cross-validation
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2)
    train_id, test_id = next(sss.split(train_image, label))
    tra_image, tra_label = train_image[train_id], label[train_id]
    val_image, val_label = train_image[test_id], label[test_id]
    return (tra_image, tra_label), (val_image, val_label)

if __name__ == '__main__':
    (tra_image, tra_label), (val_image, val_label) = load_train_data()
    onehot_tra_label = to_categorical(tra_label)
    onehot_val_label = to_categorical(val_label)

    KerasCNNmodel = Sequential()
    KerasCNNmodel.add(Convolution2D(input_shape=(512, 512, 1), filters=10, kernel_size=5, activation='relu'))
    KerasCNNmodel.add(MaxPooling2D(pool_size=(4,4)))

    KerasCNNmodel.add(Convolution2D(filters=20, kernel_size=3, activation='relu'))
    KerasCNNmodel.add(MaxPooling2D(pool_size=(2,2)))
    KerasCNNmodel.add(Convolution2D(filters=40, kernel_size=3, activation='relu'))
    KerasCNNmodel.add(MaxPooling2D(pool_size=(1,1)))

    KerasCNNmodel.add(Flatten())

    KerasCNNmodel.add(Dense(256, activation='relu'))
    KerasCNNmodel.add(Dropout(0.5))
    KerasCNNmodel.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    #rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    KerasCNNmodel.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    history=KerasCNNmodel.fit(tra_image, onehot_tra_label, epochs=10, validation_data=(val_image, onehot_val_label), batch_size=20)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    KerasCNNmodel.save('urban_sound.h5')
    del KerasCNNmodel