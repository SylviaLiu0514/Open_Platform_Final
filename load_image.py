import numpy as np
import cv2

def Load_image(ID, image_length = 512):
    img = cv2.imread('spectrogram/' + str(ID) + '.png')
    cv2.imshow("img",img)

Load_image('1')
cv2.waitKey (0)