# -*- coding: utf-8 -*-
"""
Created on Tue 29 Dec 2020

@author: Pierre Barbat Maximilien Cetre Thomas Corcoral
"""

"""
The purpose of this file is to create the window that will allow the user to use
the project to identify, play, or see the different spectrograms of a sound 
"""

##########################################
# Importation
##########################################

import format_data as fd
import cnn_model as cnn
import prediction as pred
import generate_csv as gc
import tkinter as tk
from tkinter import filedialog
import os
import shutil
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read
from playsound import playsound
from keras.models import model_from_json
from pygame import mixer
from pydub import AudioSegment

##########################################
# Variables globales
##########################################

WIDTH = 1300
HEIGHT = 740
BACKGROUND_MENU = '#445976'
BACKGROUND_TITLE = '#2f3d51'
BACKGROUND_SOUND = '#59759c'
LENGTH_BUT = 40
WIDTH_BUT = 175
global data_path
global path_csv
global test_path
global model

##########################################
# Définition des classes
##########################################

"""
This part is for the appearance of all the element inside the window, like the size of it, where the buttons are
For instance, once the accuracy percentage reaches a certain threshold, it will change color to indicate ifyou should trust or not the prediction
"""


class Menu:
    def __init__(self, can, quit_pic, run_pic, folder_pic, open_but, run_but, quit_but, csv_pic, open_csv_but,
                 format_data_but, format_pic, generate_csv_but,
                 save_csv_img):
        self.can = can
        self.quit_pic = quit_pic
        self.run_pic = run_pic
        self.folder_pic = folder_pic
        self.open_but = open_but
        self.run_but = run_but
        self.quit_but = quit_but
        self.csv_pic = csv_pic
        self.open_csv_but = open_csv_but
        self.format_data_but = format_data_but
        self.format_pic = format_pic
        self.generate_csv_but = generate_csv_but
        self.save_csv_img = save_csv_img

    def config(self):
        self.run_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                            relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")
        self.generate_csv_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                                     relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")
        self.open_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                             relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")
        self.open_csv_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                                 relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")
        self.format_data_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                                    relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")
        self.quit_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                             relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")

    def display(self):
        self.open_but.place(x=2, y=52)
        self.generate_csv_but.place(x=2, y=96)
        self.open_csv_but.place(x=2, y=140)
        self.format_data_but.place(x=2, y=184)
        self.run_but.place(x=2, y=228)
        self.quit_but.place(x=2, y=HEIGHT - 45)
        self.can.place(x=-1, y=0)


class Header:
    def __init__(self, can, icon):
        self.can = can
        self.icon = icon

    def creation(self):
        self.can.create_image(30, 25, image=self.icon)
        self.can.create_text(100, 25, font=("Courrier", 16), fill='white', text="Projet L3")

    def display(self):
        self.can.place(x=2, y=2)


class InfosMenu:
    def __init__(self, can_menu, text, label, epoch, label_epoch, spec, mfcc_choice, spec_choice, ratio,
                 ratio_spinbox, rs, rs_spinbox, label_rs, label_ratio, save_model_but, name_model, name_entry,
                 save_data_but, name_data, name_data_entry):
        self.can_menu = can_menu
        self.text = text
        self.label = label
        self.epoch = epoch
        self.label_epoch = label_epoch
        self.spec = spec
        self.mfcc_choice = mfcc_choice
        self.spec_choice = spec_choice
        self.ratio = ratio
        self.ratio_spinbox = ratio_spinbox
        self.rs = rs
        self.rs_spinbox = rs_spinbox
        self.label_rs = label_rs
        self.label_ratio = label_ratio
        self.save_model_but = save_model_but
        self.name_model = name_model
        self.name_entry = name_entry
        self.save_data_but = save_data_but
        self.name_data = name_data
        self.name_data_entry = name_data_entry

    def change_percent(self, new):
        if type(new) is tuple:
            new = int(new[0])
        if type(new) is float:
            new = int(new)
        self.text.set(str(new) + " %")
        if new < 40:
            self.label.config(fg="red")
        elif new < 60:
            self.label.config(fg="orange")
        elif new < 80:
            self.label.config(fg="yellow")
        else:
            self.label.config(fg="green")

    def get_epochs(self):
        return self.epoch.get()

    def get_spec(self):
        return bool(self.spec.get())

    def get_rs(self):
        return self.rs.get()

    def get_ratio(self):
        return float(self.ratio.get())

    def get_save_name(self):
        return self.name_model.get()

    def get_data_name(self):
        return self.name_data.get()

    def display(self):
        self.can_menu.place(x=2, y=270)
        self.label.place(x=75, y=280)
        self.label_epoch.place(x=35, y=318)
        self.epoch.place(x=100, y=320)
        self.mfcc_choice.place(x=35, y=350)
        self.spec_choice.place(x=35, y=375)
        self.label_ratio.place(x=35, y=405)
        self.ratio_spinbox.place(x=100, y=405)
        self.label_rs.place(x=35, y=435)
        self.rs_spinbox.place(x=100, y=435)
        self.save_data_but.place(x=135, y=615)
        self.name_data_entry.place(x=10, y=620)
        self.save_model_but.place(x=135, y=650)
        self.name_entry.place(x=10, y=655)


class Footer:
    def __init__(self, can, but):
        self.can = can
        self.but = but

    def creation(self):
        self.can.create_text(20, 20, font=("Courrier", 12), fill='white', text="v 0.2")

    def display(self):
        self.but.place(x=WIDTH - 40, y=HEIGHT - 33)
        self.can.place(x=WIDTH_BUT + 5, y=HEIGHT - LENGTH_BUT - 2)


class AffichageSon:
    def __init__(self, can, show_audio, show_spec, show_mfcc, play_btn, wav_pic, process_pic, open_test_but,
                 run_test_but):
        self.can = can
        self.show_audio = show_audio
        self.show_spec = show_spec
        self.show_mfcc = show_mfcc
        self.play_btn = play_btn
        self.wav_pic = wav_pic
        self.process_pic = process_pic
        self.open_test_but = open_test_but
        self.run_test_but = run_test_but

    def config(self):
        self.open_test_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bd=1, highlightthickness=0,
                                  relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")
        self.run_test_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bd=1, highlightthickness=0,
                                 relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")

    def creation(self):
        self.can.create_text(WIDTH_BUT + 5, HEIGHT / 2 - 50, font=("Courrier", 12), fill='white', text="v 0.1")

    def display(self):
        self.can.place(x=WIDTH_BUT + 5, y=HEIGHT / 2 + 10)
        self.show_audio.place(x=300, y=HEIGHT / 2 + 180)
        self.show_spec.place(x=650, y=HEIGHT / 2 + 180)
        self.show_mfcc.place(x=1100, y=HEIGHT / 2 + 180)
        self.play_btn.place(x=680, y=HEIGHT / 2 + 130)
        self.open_test_but.place(x=435, y=HEIGHT / 2 + 210)
        self.run_test_but.place(x=880, y=HEIGHT / 2 + 210)


class AffichageRes:
    def __init__(self, can, prediction_label, prediction, resultats, resultats_label):
        self.can = can
        self.prediction_label = prediction_label
        self.prediction = prediction
        self.resultats = resultats
        self.resultats_label = resultats_label

    def new_prediction(self, new):
        self.prediction.set("Le modèle pense qu'il s'agit de : " + new)

    def new_res(self, new):
        self.resultats.set(new)

    def display(self):
        self.prediction_label.place(x=450, y=HEIGHT / 2 - 300)
        self.resultats_label.place(x=450, y=HEIGHT / 2 - 250)


class RecapSelect:
    def __init__(self, can, data_path_label, data_path_var, csv_path_label, csv_path_var):
        self.can = can
        self.data_path_label = data_path_label
        self.data_path_var = data_path_var
        self.csv_path_label = csv_path_label
        self.csv_path_var = csv_path_var

    def update_data_path(self, new):
        self.data_path_var.set("Data path : " + new)

    def update_csv_path(self, new):
        self.csv_path_var.set("CSV path : " + new)

    def display(self):
        self.can.place(x=WIDTH_BUT + 20, y=HEIGHT / 2 + 265)
        self.data_path_label.place(x=WIDTH_BUT + 40, y=HEIGHT / 2 + 270)
        self.csv_path_label.place(x=WIDTH_BUT + 40, y=HEIGHT / 2 + 295)


##########################################
# Fonctions graphique
##########################################

# This function is used to initialize the window, and lock its size
def init_window():
    w = tk.Tk()
    w.title("Projet L3")
    w.geometry(str(WIDTH) + "x" + str(HEIGHT))
    w.minsize(WIDTH, HEIGHT)
    w.maxsize(WIDTH, HEIGHT)
    w.tk.call('wm', 'iconphoto', w, tk.PhotoImage(file="../img/logo.png"))
    return w


# This function prints the current paths to the csv and data for the test
def test():
    print("path csv : " + path_csv)
    print("path data : " + data_path)


# This function creates the buttons needed to indicate the paths of the different data needed / processing the data
def init_menu():
    folder_img = tk.PhotoImage(file='../img/folder.png').subsample(14, 14)
    save_csv_img = tk.PhotoImage(file='../img/save_csv.png').subsample(14, 14)
    run_pic = tk.PhotoImage(file='../img/funnel.png').subsample(14, 14)
    csv_pic = tk.PhotoImage(file='../img/csv.png').subsample(14, 14)
    format_pic = tk.PhotoImage(file='../img/format.png').subsample(14, 14)
    quit_pic = tk.PhotoImage(file='../img/leave.png').subsample(14, 14)
    can_menu = tk.Canvas(window, width=0, height=0)
    open_train_but = tk.Button(window, image=folder_img, text="  Data Path", font=("Courrier", 14), fg='black',
                               compound='left', command=choose_dir_data)
    generate_csv_but = tk.Button(window, image=save_csv_img, text="  Generer .CSV", font=("Courrier", 14), fg='black',
                                 compound='left', command=generate_csv)
    open_csv_but = tk.Button(window, image=csv_pic, text="  CSV Path", font=("Courrier", 14), fg='black',
                             compound='left', command=choose_path_csv)
    format_data_but = tk.Button(window, image=format_pic, text="  Format data", font=("Courrier", 14), fg='black',
                                compound='left', command=format_data)
    run_train_but = tk.Button(window, image=run_pic, text="  Run Train", font=("Courrier", 14), fg='black',
                              compound='left', command=run_model)
    quit_but = tk.Button(window, image=quit_pic, text="  Quitter", font=("Courrier", 14), fg='black', compound='left',
                         command=leave)
    win_menu = Menu(can_menu, quit_pic, run_pic, folder_img, open_train_but, run_train_but, quit_but, csv_pic,
                    open_csv_but, format_data_but, format_pic, generate_csv_but, save_csv_img)
    return win_menu


# This is used to initialize the header for the window
def init_header():
    icon = tk.PhotoImage(file="../img/logo.png").subsample(12, 12)
    can_head = tk.Canvas(window, width=WIDTH_BUT + 4, height=HEIGHT / 15, bd=0, highlightthickness=0, relief='ridge',
                         bg=BACKGROUND_TITLE)
    win_header = Header(can_head, icon)
    return win_header


# This is used to indicate all the needed informations to write the text on the buttons
def init_infos_menu():
    can_menu = tk.Canvas(window, width=179, height=425, bg=BACKGROUND_TITLE, bd=0, highlightthickness=0, relief='ridge')
    text = tk.StringVar()
    text.set("- %")
    label = tk.Label(window, textvariable=text, font=("Courrier", 14), bg=BACKGROUND_TITLE)
    label_epoch = tk.Label(window, text="Epochs ", font=("Courrier", 10), bg=BACKGROUND_TITLE)
    val = tk.StringVar()
    val.set(10)
    epoch = tk.Spinbox(window, from_=10, to=1000, increment=5, textvariable=val, width=5)
    spec = tk.IntVar()
    mfcc_choice = tk.Radiobutton(window, text="MFCC", variable=spec, value=0, bg=BACKGROUND_TITLE)
    mfcc_choice.select()
    spec_choice = tk.Radiobutton(window, text="SPECTROGRAMME", variable=spec, value=1, bg=BACKGROUND_TITLE)
    label_ratio = tk.Label(window, text="Ratio ", font=("Courrier", 10), bg=BACKGROUND_TITLE)
    ratio = tk.StringVar()
    ratio.set(10)
    ratio_spinbox = tk.Spinbox(window, from_=0, to=1, increment=.05, textvariable=ratio, width=5)
    label_rs = tk.Label(window, text="RS ", font=("Courrier", 10), bg=BACKGROUND_TITLE)
    rs = tk.StringVar()
    rs.set(10)
    rs_spinbox = tk.Spinbox(window, from_=0, to=100, increment=1, textvariable=rs, width=5)
    save_data_but = tk.Button(window, text="save", font=("Courrier", 11), fg='black', command=save_as_data_format)
    name_data = tk.StringVar(value='data')
    name_data_entry = tk.Entry(window, textvariable=name_data)
    save_model_but = tk.Button(window, text="save", font=("Courrier", 11), fg='black', command=save_as_model)
    name_model = tk.StringVar(value='model')
    name_entry = tk.Entry(window, textvariable=name_model)
    infos_menu = InfosMenu(can_menu, text, label, epoch, label_epoch, spec, mfcc_choice, spec_choice, ratio,
                           ratio_spinbox, rs, rs_spinbox, label_rs, label_ratio, save_model_but, name_model, name_entry,
                           save_data_but, name_data, name_data_entry)
    return infos_menu


# This is used to initialize the footer for the window
def init_footer():
    can_footer = tk.Canvas(window, width=WIDTH - WIDTH_BUT, height=LENGTH_BUT, bg=BACKGROUND_TITLE, bd=0,
                           highlightthickness=0, relief='ridge')
    web_button = tk.Button(window, text="Aide", font=("Courrier", 10), bd=0, highlightthickness=0, relief='ridge',
                           command=show_aide)
    foot = Footer(can_footer, web_button)
    return foot


# This is used to return the results of the prediction
def init_resultats():
    can = tk.Canvas(window, width=WIDTH - WIDTH_BUT, height=HEIGHT / 2 - 50, bg=BACKGROUND_SOUND, bd=0,
                    highlightthickness=0, relief='ridge')
    prediction = tk.StringVar()
    prediction.set("")
    prediction_label = tk.Label(window, textvariable=prediction, font=("Courrier", 16), bg=BACKGROUND_SOUND)
    resultats = tk.StringVar()
    resultats.set("")
    resultats_label = tk.Label(window, textvariable=resultats, font=("Courrier", 14), bg=BACKGROUND_SOUND)
    result = AffichageRes(can, prediction_label, prediction, resultats, resultats_label)
    return result


# This is used to create the buttons used to see the graphic representations of a sound
def init_sons():
    wav_pic = tk.PhotoImage(file='../img/wav.png').subsample(14, 14)
    process_pic = tk.PhotoImage(file='../img/process.png').subsample(14, 14)
    can = tk.Canvas(window, width=0, height=0, bg=BACKGROUND_SOUND, bd=0,
                    highlightthickness=0, relief='ridge')
    show_audio = tk.Button(window, text="Voir audio", font=("Courrier", 14), fg='black',
                           compound='left', command=show_audio_representation)
    show_spec = tk.Button(window, text="Voir spectrogramme", font=("Courrier", 14), fg='black',
                          compound='left', command=show_spectrogramme)
    show_mfcc = tk.Button(window, text="Voir mfcc", font=("Courrier", 14), fg='black',
                          compound='left', command=show_mfccs)
    play_btn = tk.Button(window, text='Play Test File', font=("Courrier", 14), command=lambda: run_test_audio())
    open_test_but = tk.Button(window, image=wav_pic, text="  Test Path", font=("Courrier", 14), fg='black',
                              compound='left', command=choose_test_path)
    run_test_but = tk.Button(window, image=process_pic, text="  Run Test", font=("Courrier", 14), fg='black',
                             compound='left', command=predict)
    show_son = AffichageSon(can, show_audio, show_spec, show_mfcc, play_btn, wav_pic, process_pic, open_test_but,
                            run_test_but)
    return show_son


# This is used to initialize the model using model.json and the obtained accuracy
def init_model():
    try:
        # load json and create model
        file = open("../local_saves/model/model.json", 'r')
        file_acc = open("../local_saves/accuracy.txt", 'r')
    except IOError:
        print("File not accessible")
        return None
    model_json = file.read()
    acc_str = file_acc.read()
    if acc_str != '':
        accuracy = int(float(acc_str))
        menu_infos.change_percent(accuracy)
        model_local = model_from_json(model_json)
        file.close()
        # load weights
        model_local.load_weights("../local_saves/model/model.h5")
        return model_local
    return None


def init_recap_selec():
    can = tk.Canvas(window, width=WIDTH - WIDTH_BUT - 40, height=60, bg="white", bd=0, highlightthickness=0,
                    relief='ridge')
    data_path_var = tk.StringVar()
    data_path_var.set("Data path :")
    csv_path_var = tk.StringVar()
    csv_path_var.set("CSV path :")
    data_path_label = tk.Label(window, textvariable=data_path_var, font=("Courrier", 11))
    csv_path_label = tk.Label(window, textvariable=csv_path_var, font=("Courrier", 11))
    recapit = RecapSelect(can, data_path_label, data_path_var, csv_path_label, csv_path_var)
    return recapit


##########################################
# Fonctions internes
##########################################


# This is used to play the sound so you can hear it. If it's an mp3, it will use pygame to play it,
# otherwise, it will use playsound
def run_test_audio():
    if test_path != '':
        if test_path.endswith('.mp3'):
            mixer.init()
            mixer.music.load(test_path)
            mixer.music.play()
        else:
            playsound(test_path)


# This is used to quit the application
def leave():
    window.destroy()


# This allows the estimated accuracy to be changed using the value in parameters
def change(new):
    menu_infos.change_percent(new)


# This is used to indicate the path to the dataset
def choose_dir_data():
    global data_path
    new = filedialog.askdirectory(initialdir="./", title="Selectionnez votre dataset")
    if new != '':
        data_path = new
        recap.update_data_path(data_path)


# This is used to indicate the path to the .csv
def choose_path_csv():
    global path_csv
    new = filedialog.askopenfilename(initialdir="./", title="Selectionnez votre fichier .csv",
                                     filetypes=(("csv  files", "*.csv"), ("all files", "*.*")))
    if new != '':
        path_csv = new
        recap.update_csv_path(path_csv)


# This is used to indicate the path to the tested file
def choose_test_path():
    global test_path
    new = filedialog.askopenfilename(initialdir="./", title="Selectionnez votre fichier son",
                                     filetypes=((".wav, .mp3", "*.wav, *.mp3"), ("all files", "*.*")))
    if new != '':
        test_path = new


# This is used to clear the paths to the different
def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            return -1
    return 0


# This is used to format the data using the different paths indicated
def format_data():
    if data_path == "":
        print("Erreur : Vous devez renseigner le chemin de votre dataset")
        return
    if path_csv == "":
        print("Erreur : Vous devez renseigner le chemin de votre fichier csv")
        return
    if clear_folder("../local_saves") == -1:
        print("Erreur : Le dossier ../local_saves n'a pas pu être nettoye")
        return
    print("Lancement pour le dossier : " + data_path)
    print("Et le fichier CSV : " + path_csv)
    # if fd.conv_data(data_path, path_csv, ratio=menu_infos.get_ratio(), rs=menu_infos.get_rs(),
    #                 spec=menu_infos.get_spec()) == -1:
    if fd.conv_data(data_path, path_csv, spec=menu_infos.get_spec()) == -1:
        print("Erreur : Un des fichier indiqué n'existe pas")
        return


# This is used to run the model using cnn_model.py after formatting the data
def run_model():
    global model
    if not (os.path.isfile('../local_saves/data_format/test_audio.npy')):
        print("Il manque le fichier test_audio.npy essayez de relancer le formatage des fichiers")
        return
    if not (os.path.isfile('../local_saves/data_format/train_audio.npy')):
        print("Il manque le fichier train_audio.npy essayez de relancer le formatage des fichiers")
        return
    if not (os.path.isfile('../local_saves/data_format/test_labels.npy')):
        print("Il manque le fichier test_labels.npy essayez de relancer le formatage des fichiers")
        return
    if not (os.path.isfile('../local_saves/data_format/train_labels.npy')):
        print("Il manque le fichier train_labels.npy essayez de relancer le formatage des fichiers")
        return
    accuracy, model = cnn.run_model(int(menu_infos.get_epochs()))
    menu_infos.change_percent(accuracy)


# This is used to get the prediction of what bird / bat species the test sound corresponds to, and then print it
def predict():
    global model, path_csv
    if test_path == "":
        print("Erreur : Vous devez selectionner un fichier audio .wav en cliquant sur le bouton Test Path")
        return
    if not model:
        accuracy, model = cnn.run_model()
    # pred.create_prediction()

    mfcc = True

    if menu_infos.get_spec() == 1:
        mfcc = False

    resultat, le, predicted_proba = pred.print_prediction(test_path, model, mfcc)
    res.new_prediction(resultat)
    all_prob = ""
    for i in range(len(predicted_proba)):
        category = le.inverse_transform(np.array([i]))
        current_classe = category[0]
        if type(current_classe) is tuple:
            current_classe = str(current_classe[0])
        prob = int(float(predicted_proba[i]) * 100)
        all_prob += current_classe + " : " + str(prob) + " %\n"
    res.new_res(all_prob)


# This is used to show the spectrogram corresponding to the sound that is being tested
def show_spectrogramme():
    if not (os.path.isfile(test_path)):
        return
    audio, sample_rate = librosa.load(test_path, res_type='kaiser_fast')
    spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128,
                                          fmax=11000, power=0.5)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spec, x_axis='time')
    plt.colorbar()
    plt.title('SPECTROGRAMME')
    plt.tight_layout()
    plt.show()


# This is used to show the mfcc corresponding to the sound that is being tested
def show_mfccs():
    if not (os.path.isfile(test_path)):
        return
    audio, sample_rate = librosa.load(test_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=100, hop_length=1024, htk=True)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()


# This is used to show the audio representation corresponding to the sound that is being tested
def show_audio_representation():
    if not (os.path.isfile(test_path)):
        return
    if test_path.endswith('.mp3'):
        sound = AudioSegment.from_mp3(test_path)
        dst = '../local_saves/current.wav'
        sound.export(dst, format="wav")
    else:
        dst = test_path
    samplerate, data = read(dst)
    duration = len(data) / samplerate
    time = np.arange(0, duration, 1 / samplerate)  # time vector
    plt.plot(time, data)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Représentation de l audio')
    plt.show()


# This is used save the current model in the "local save" folder to be able to use it later without re-creating every
# step
def save_as_model():
    if not os.path.isdir("../local_saves/model"):
        return
    if os.listdir('../local_saves/model') == 0:
        return
    print("save as")
    save_model_path = filedialog.askdirectory(initialdir="./",
                                              title="Choose a path to save your model")
    print(save_model_path)
    val_name = menu_infos.get_save_name()
    if val_name == '':
        shutil.make_archive(save_model_path + "/my_model", "zip", "../local_saves/model")
    else:
        shutil.make_archive(save_model_path + "/" + val_name, "zip", "../local_saves/model")
    print("Copie terminée")


# This is used to save the data for later
def save_as_data_format():
    if not os.path.isdir("../local_saves/data_format"):
        return
    if os.listdir('../local_saves/data_format') == 0:
        return
    print("save as")
    save_model_path = filedialog.askdirectory(initialdir="./",
                                              title="Choose a path to save your data")
    print(save_model_path)
    val_name = menu_infos.get_data_name()
    if val_name == '':
        shutil.make_archive(save_model_path + "/my_data", "zip", "../local_saves/data_format")
    else:
        shutil.make_archive(save_model_path + "/" + val_name, "zip", "../local_saves/data_format")
    print("Copy finish")


# This is used to generate a .csv (Spreadsheet) using the Data set indicated
def generate_csv():
    print("generate csv")
    if data_path == "":
        print("Erreur : Vous devez renseigner le chemin de votre dataset")
        return
    gc.generate(data_path)


##########################################
# Aide
##########################################

"""
This class is used to print the help / instructions should the user press the help button in the bottom right corner
"""


class Aide:
    def __init__(self, main_can):
        self.main_can = main_can

    def creation(self):
        f = open("../aide.txt", "r")
        self.main_can.create_text(WIDTH / 2, HEIGHT / 2, font=("Courrier", 12), fill='black', text=f.read())

    def display(self):
        self.main_can.place(x=0, y=0)


# Is used to create the help window
def init_aide(win):
    main_can = tk.Canvas(win, width=WIDTH, height=HEIGHT, bd=0,
                         highlightthickness=0, relief='ridge')
    aide = Aide(main_can)
    return aide


# Is used to show aide.txt
def show_aide():
    win = tk.Toplevel(window)
    win.title("Aide Projet L3")
    local_width = WIDTH
    local_height = HEIGHT
    win.geometry(str(local_width) + "x" + str(local_height))
    win.minsize(local_width, local_height)
    win.maxsize(local_width, local_height)
    win.tk.call('wm', 'iconphoto', win, tk.PhotoImage(file="../img/logo.png"))
    aide = init_aide(win)
    aide.creation()
    aide.display()


##########################################
# Main
##########################################


if __name__ == "__main__":
    # Création de la fenêtre
    window = init_window()

    # Initialisation du fond d'écran de l'application
    background_image = tk.PhotoImage(file='../img/background.png')
    background_label = tk.Label(window, image=background_image)
    background_label.place(x=0, y=0)

    # Création du menu
    menu = init_menu()
    menu.config()
    menu.display()

    # Création du Header
    header = init_header()
    header.creation()
    header.display()

    # Création de l'emplacement pour les options
    menu_infos = init_infos_menu()
    menu_infos.display()

    # Création du footer
    footer = init_footer()
    footer.creation()
    footer.display()

    # Création de l'affichage du résultat
    res = init_resultats()
    res.display()

    # Création de l'affichage du son
    son = init_sons()
    son.config()
    son.display()

    # Création des informations concernant les chemins
    recap = init_recap_selec()
    recap.display()

    # Initialisation des variables
    data_path = ""
    path_csv = ""
    test_path = ""
    model = init_model()

    window.mainloop()
