import tkinter as tk
from tkinter import messagebox  # import this to fix messagebox error
from tkinter import filedialog
from tkinter import StringVar, IntVar
import pickle
from PIL import Image, ImageTk
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.misc

window = tk.Tk()
    #window.geometry('450x300')
def Login():
    #window = tk.Toplevel(window)
    global window
    window.destroy()
    window = tk.Tk()
    window.title('LosinView')
    window.geometry('450x300')
    window.protocol("WM_DELETE_WINDOW", Quit)
    # welcome image
    load = Image.open('welcome.png');
    load = load.resize((450, 130))
    render = ImageTk.PhotoImage(load);
    img = tk.Label( image = render);
    img.image = render;
    img.place(x = 0, y = 0);  #coord(0,0)

    # user information
    tk.Label(window, text='User name: ').place(x=50, y= 150)
    tk.Label(window, text='Password: ').place(x=50, y= 190)

    var_usr_name = tk.StringVar()
    var_usr_name.set('example@python.com')
    entry_usr_name = tk.Entry(window, textvariable=var_usr_name)
    entry_usr_name.place(x=160, y=150)
    var_usr_pwd = tk.StringVar()
    entry_usr_pwd = tk.Entry(window, textvariable=var_usr_pwd, show='*')
    entry_usr_pwd.place(x=160, y=190)

    def usr_login():
        usr_name = var_usr_name.get()
        usr_pwd = var_usr_pwd.get()
        #check usrs_info.pickle
        try:
            with open('usrs_info.pickle', 'rb') as usr_file:
                usrs_info = pickle.load(usr_file)
        except FileNotFoundError:
            with open('usrs_info.pickle', 'wb') as usr_file:
                usrs_info = {'admin': 'admin'}
                pickle.dump(usrs_info, usr_file)
        if usr_name in usrs_info:
            if usr_pwd == usrs_info[usr_name]:
                MainView()
            else:
                tk.messagebox.showerror(message='Error, your password is wrong, try again.')
        else:
            if tk.messagebox.askyesno('Welcome','You have not signed up yet. Sign up today?'):
                usr_sign_up()

    def usr_sign_up():
        def sign_to_Mofan_Python():
            np = new_pwd.get()
            npf = new_pwd_confirm.get()
            nn = new_name.get()
            with open('usrs_info.pickle', 'rb') as usr_file:
                exist_usr_info = pickle.load(usr_file)
            if np != npf:
                tk.messagebox.showerror('Error', 'Password and confirm password must be the same!')
            elif nn in exist_usr_info:
                tk.messagebox.showerror('Error', 'The user has already signed up!')
            else:
                exist_usr_info[nn] = np
                with open('usrs_info.pickle', 'wb') as usr_file:
                    pickle.dump(exist_usr_info, usr_file)
                tk.messagebox.showinfo('Welcome', 'You have successfully signed up!')
                window_sign_up.destroy()
        #create a new window top to origin
        window_sign_up = tk.Toplevel(window)
        window_sign_up.geometry('350x200')
        window_sign_up.title('Sign up window')
        #check usrs_info.pickle
        try:
            with open('usrs_info.pickle', 'rb') as usr_file:
                usrs_info = pickle.load(usr_file)
        except FileNotFoundError:
            with open('usrs_info.pickle', 'wb') as usr_file:
                usrs_info = {'admin': 'admin'}
                pickle.dump(usrs_info, usr_file)
        new_name = tk.StringVar()
        new_name.set('example@python.com')
        tk.Label(window_sign_up, text='User name: ').place(x=10, y= 10)
        entry_new_name = tk.Entry(window_sign_up, textvariable=new_name)
        entry_new_name.place(x=150, y=10)

        new_pwd = tk.StringVar()
        tk.Label(window_sign_up, text='Password: ').place(x=10, y=50)
        entry_usr_pwd = tk.Entry(window_sign_up, textvariable=new_pwd, show='*')
        entry_usr_pwd.place(x=150, y=50)

        new_pwd_confirm = tk.StringVar()
        tk.Label(window_sign_up, text='Confirm password: ').place(x=10, y= 90)
        entry_usr_pwd_confirm = tk.Entry(window_sign_up, textvariable=new_pwd_confirm, show='*')
        entry_usr_pwd_confirm.place(x=150, y=90)

        btn_comfirm_sign_up = tk.Button(window_sign_up, text='Sign up', command=sign_to_Mofan_Python)
        btn_comfirm_sign_up.place(x=150, y=130)

    # login and sign up button
    btn_login = tk.Button(window, text='Login', command=usr_login)
    btn_login.place(x=170, y=230)
    btn_sign_up = tk.Button(window, text='SignUp', command=usr_sign_up)
    btn_sign_up.place(x=270, y=230)
def MainView():
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
        del filename, name, clip, sample_rate, fig, ax
    def resize_image(image, image_length = 512):
        # Find deflation_rate for narrow the image
        deflation_rate = image_length / float(image.size[max((0, 1), key=lambda i: image.size[i])])
        image.resize((int(image.size[0] * deflation_rate), int(image.size[1] * deflation_rate)))
        return image.resize((int(image.size[0] * deflation_rate), int(image.size[1] * deflation_rate)))
    def load_images(filenames,image_length=512):
        output_image = np.empty((len(filenames), 512, 512, 1))
        # i -> the i th of the ID set, image_ID -> the name of the image
        for i in range(len(filenames)):
            # Turn the image into an array
            image=img_to_array(resize_image(load_img(filenames[i], color_mode = "grayscale"),image_length=512))
            # Get image height and width
            # Place the image at the center of the image
            h1 = int((512 - image.shape[0] ) / 2)
            h2 = h1 + image.shape[0]
            w1 = int((512 - image.shape[1]) / 2)
            w2 = w1 + image.shape[1]
            # Insert into image matrix
            output_image[i, h1:h2, w1:w2, 0:1] = image
            # Scale the array values so they are between 0 and 1
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
    def choose_image():
            filename = filedialog.askopenfilename(parent=window, initialdir="C:/",title='Choose an image.')
            if "wav" in filename:
                try:
                    imgname = filename.replace('wav','png')
                    create_spectrogram(filename,imgname)
                    filename = imgname
                except:
                    pass
            try:
                testdata = load_images([filename])
            #not .png or .wav
            except:
                print('file not found!')
                text = 'result: file not found!'
                tk.Label(window, text=text.ljust(20, ' ')).place(x=50, y= 320)
            else:
                result=decode(model.predict(testdata))
                print(result[0])
                text = 'result: '+result[0]
                tk.Label(window, text=text.ljust(20, ' ')).place(x=50, y= 320)
                load = Image.open('train/wav.png') if "wav" in filename else Image.open(filename)
                load = load.resize((415, 280))
                render = ImageTk.PhotoImage(load);
                #標籤可以是文字或圖片
                img = tk.Label( image = render);
                img.image = render;
                img.place(x = 15, y = 15);  #將影像放入視窗裡，座標為(50，50);
    def Logout():
        if tk.messagebox.askyesno('Logout?','Are you sure want to logout?'):
            Login()
    global window
    window.destroy()
    window = tk.Tk()
    window.geometry('450x400')
    window.title('MainView')
    window.protocol("WM_DELETE_WINDOW", Quit)
    btn_login = tk.Button(window, text='Choose', command=choose_image)
    btn_login.place(x=150, y=350)
    btn_login = tk.Button(window, text='Logout', command=Logout)
    btn_login.place(x=240, y=350)

def Quit():
    if tk.messagebox.askyesno('Quit?','Are you sure want to quit?'):
        window.destroy()
        #close function create_spectrogram() create plt
        plt.close()
        plt.close('all')


if __name__ == "__main__":
    model=load_model('urban_sound.h5')
    model.predict(np.empty((1, 512, 512, 1)))
    print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nInitial successful')
    Login()
    window.protocol("WM_DELETE_WINDOW", Quit)
    window.mainloop()
