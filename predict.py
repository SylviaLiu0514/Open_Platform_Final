from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.misc

def create_spectrogram(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y=clip, sr=sample_rate), ref=np.max))
    plt.savefig(name, dpi=50, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename, name, clip, sample_rate, fig, ax
def resize_image(image, image_length = 512):
    # Find deflation_rate for narrow the image
    deflation_rate = image_length / float(image.size[max((0, 1), key=lambda i: image.size[i])])
    image.resize((int(image.size[0] * deflation_rate), int(image.size[1] * deflation_rate)))
    return image.resize((int(image.size[0] * deflation_rate), int(image.size[1] * deflation_rate)))
def load_images(filenames,image_length=512):
    output_image = np.empty((len(filenames), 512, 512, 1))
    for i in range(len(filenames)):
        image=img_to_array(resize_image(load_img(filenames[i], color_mode = "grayscale"),image_length=512))
        h1 = int((512 - image.shape[0] ) / 2)
        h2 = h1 + image.shape[0]
        w1 = int((512 - image.shape[1]) / 2)
        w2 = w1 + image.shape[1]
        output_image[i, h1:h2, w1:w2, 0:1] = image
    return np.around(output_image / 255.0)
def decode(arr):
    result=[]
    labels=['air_conditioner','car_horn','children_playing','dog_bark','drilling','engine_idling','gun_shot','jackhammer','siren','street_music','unknow']
    for a in arr:
        find=False
        for i in range(10):
            if a[i]>0.9:
                result.append(labels[i])
                find=True
        if not find:
            result.append(labels[10])
    return result

if __name__ == '__main__':
    model=load_model('urban_sound.h5')
    model.predict(np.empty((1, 512, 512, 1)))
    print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nInitial success')
    while True:
        filename=input('File name?')
        if filename=="":
            print('Closing program..')
            break
        if filename.find('wav')!=-1:
            try:
                imgname=filename.split('.wav')[0]+".png"
                create_spectrogram(filename,imgname)
                filename=imgname
            except:
                pass
        try:
            testdata=load_images([filename])
        except:
            print('file not found!')
        else:
            result=decode(model.predict(testdata))
            print(result[0])
