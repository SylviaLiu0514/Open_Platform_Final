import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image

def resize_image(image, image_length = 512):
    # Find deflation_rate for narrow the image
    deflation_rate = image_length / float(image.size[max((0, 1), key=lambda i: image.size[i])])
    image.resize((int(image.size[0] * deflation_rate), int(image.size[1] * deflation_rate)))
    return image.resize((int(image.size[0] * deflation_rate), int(image.size[1] * deflation_rate)))

def Load_image(ID, image_length = 512):
    # i -> the i th of the ID set, image_ID -> the name of the image
    image = img_to_array(resize_image(load_img('spectrogram/' + str(ID) + '.png', color_mode = "grayscale"),
                                                    image_length=image_length))
    return image

print(Load_image('1'))