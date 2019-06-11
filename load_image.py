import numpy as np
import cv2
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image

def resize_image(image, image_length = 512):
    deflation_rate = image_length / float(image.size[max((0, 1), key=lambda i: image.size[i])])
    image.resize((int(image.size[0] * deflation_rate), int(image.size[1] * deflation_rate)))
    return image.resize((int(image.size[0] * deflation_rate), int(image.size[1] * deflation_rate)))
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

print(Load_image('1'))