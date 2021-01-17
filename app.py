# -*- coding: utf-8 -*-
"""
Created on Tue 29 Dec 2020

@author: Pierre Barbat Maximilien Cetre Thomas Corcoral
"""

##########################################
# Importation
##########################################

import format_data as fd
import cnn_model as cnn
import prediction as pred
import tkinter as tk
from tkinter import filedialog
import os
import shutil
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read
from winsound import *

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


class Menu:
    def __init__(self, can, quit_pic, run_pic, folder_pic, open_but, run_but, open_test_but, run_test_but, quit_but,
                 wav_pic, process_pic, csv_pic, open_csv_but, format_data_but, format_pic):
        self.can = can
        self.quit_pic = quit_pic
        self.run_pic = run_pic
        self.folder_pic = folder_pic
        self.open_but = open_but
        self.run_but = run_but
        self.quit_but = quit_but
        self.open_test_but = open_test_but
        self.run_test_but = run_test_but
        self.wav_pic = wav_pic
        self.process_pic = process_pic
        self.csv_pic = csv_pic
        self.open_csv_but = open_csv_but
        self.format_data_but = format_data_but
        self.format_pic = format_pic

    def config(self):
        self.run_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                            relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")
        self.open_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                             relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")
        self.open_csv_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                                 relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")
        self.format_data_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                                    relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")
        self.quit_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                             relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")
        self.open_test_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                                  relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")
        self.run_test_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                                 relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")

    def display(self):
        self.open_but.place(x=2, y=52)
        self.open_csv_but.place(x=2, y=96)
        self.format_data_but.place(x=2, y=140)
        self.run_but.place(x=2, y=184)
        self.open_test_but.place(x=2, y=228)
        self.run_test_but.place(x=2, y=272)
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
    def __init__(self, can_menu, text, label, epoch, label_epoch, play_btn, spec, mfcc_choice, spec_choice, ratio,
                 ratio_spinbox, rs, rs_spinbox, label_rs, label_ratio):
        self.can_menu = can_menu
        self.text = text
        self.label = label
        self.epoch = epoch
        self.label_epoch = label_epoch
        self.play_btn = play_btn
        self.spec = spec
        self.mfcc_choice = mfcc_choice
        self.spec_choice = spec_choice
        self.ratio = ratio
        self.ratio_spinbox = ratio_spinbox
        self.rs = rs
        self.rs_spinbox = rs_spinbox
        self.label_rs = label_rs
        self.label_ratio = label_ratio

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

    def display(self):
        self.can_menu.place(x=2, y=316)
        self.label.place(x=75, y=340)
        self.label_epoch.place(x=35, y=378)
        self.epoch.place(x=100, y=380)
        self.play_btn.place(x=58, y=410)
        self.mfcc_choice.place(x=35, y=450)
        self.spec_choice.place(x=35, y=470)
        self.label_ratio.place(x=35, y=500)
        self.ratio_spinbox.place(x=100, y=500)
        self.label_rs.place(x=35, y=530)
        self.rs_spinbox.place(x=100, y=530)


class Footer:
    def __init__(self, can, but):
        self.can = can
        self.but = but

    def creation(self):
        self.can.create_text(20, 20, font=("Courrier", 12), fill='white', text="v 0.1")

    def display(self):
        self.but.place(x=WIDTH - 40, y=HEIGHT - 33)
        self.can.place(x=WIDTH_BUT + 5, y=HEIGHT - LENGTH_BUT - 2)


class AffichageSon:
    def __init__(self, can, show_audio, show_spec, show_mfcc):
        self.can = can
        self.show_audio = show_audio
        self.show_spec = show_spec
        self.show_mfcc = show_mfcc

    def creation(self):
        self.can.create_text(WIDTH_BUT + 5, HEIGHT / 2 - 50, font=("Courrier", 12), fill='white', text="v 0.1")

    def display(self):
        self.can.place(x=WIDTH_BUT + 5, y=HEIGHT / 2 + 10)
        self.show_audio.place(x=300, y=HEIGHT / 2 + 200)
        self.show_spec.place(x=650, y=HEIGHT / 2 + 200)
        self.show_mfcc.place(x=1100, y=HEIGHT / 2 + 200)


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


##########################################
# Fonctions graphique
##########################################


def init_window():
    w = tk.Tk()
    w.title("Projet L3")
    w.geometry(str(WIDTH) + "x" + str(HEIGHT))
    w.minsize(WIDTH, HEIGHT)
    w.maxsize(WIDTH, HEIGHT)
    w.tk.call('wm', 'iconphoto', w, tk.PhotoImage(file="./img/logo.png"))
    return w


def test():
    print("path csv : " + path_csv)
    print("path data : " + data_path)


def init_menu():
    folder_img = tk.PhotoImage(file='./img/folder.png').subsample(14, 14)
    run_pic = tk.PhotoImage(file='./img/funnel.png').subsample(14, 14)
    csv_pic = tk.PhotoImage(file='./img/csv.png').subsample(14, 14)
    format_pic = tk.PhotoImage(file='./img/format.png').subsample(14, 14)
    wav_pic = tk.PhotoImage(file='./img/wav.png').subsample(14, 14)
    process_pic = tk.PhotoImage(file='./img/process.png').subsample(14, 14)
    quit_pic = tk.PhotoImage(file='./img/leave.png').subsample(14, 14)

    can_menu = tk.Canvas(window, width=0, height=0)
    open_train_but = tk.Button(window, image=folder_img, text="  Data Path", font=("Courrier", 14), fg='black',
                               compound='left', command=choose_dir_data)
    open_csv_but = tk.Button(window, image=csv_pic, text="  CSV Path", font=("Courrier", 14), fg='black',
                             compound='left', command=choose_path_csv)
    format_data_but = tk.Button(window, image=format_pic, text="  Format data", font=("Courrier", 14), fg='black',
                                compound='left', command=format_data)
    run_train_but = tk.Button(window, image=run_pic, text="  Run Train", font=("Courrier", 14), fg='black',
                              compound='left', command=run_model)
    open_test_but = tk.Button(window, image=wav_pic, text="  Test Path", font=("Courrier", 14), fg='black',
                              compound='left', command=choose_test_path)
    run_test_but = tk.Button(window, image=process_pic, text="  Run Test", font=("Courrier", 14), fg='black',
                             compound='left', command=predict)
    quit_but = tk.Button(window, image=quit_pic, text="  Quitter", font=("Courrier", 14), fg='black', compound='left',
                         command=leave)

    win_menu = Menu(can_menu, quit_pic, run_pic, folder_img, open_train_but, run_train_but, open_test_but, run_test_but,
                    quit_but, wav_pic, process_pic, csv_pic, open_csv_but, format_data_but, format_pic)

    return win_menu


def init_header():
    icon = tk.PhotoImage(file="./img/logo.png").subsample(12, 12)
    can_head = tk.Canvas(window, width=WIDTH_BUT + 4, height=HEIGHT / 15, bd=0, highlightthickness=0, relief='ridge',
                         bg=BACKGROUND_TITLE)
    win_header = Header(can_head, icon)
    return win_header


def init_infos_menu():
    can_menu = tk.Canvas(window, width=179, height=380, bg=BACKGROUND_TITLE, bd=0, highlightthickness=0, relief='ridge')
    text = tk.StringVar()
    text.set("- %")
    label = tk.Label(window, textvariable=text, font=("Courrier", 14), bg=BACKGROUND_TITLE)
    label_epoch = tk.Label(window, text="Epochs ", font=("Courrier", 10), bg=BACKGROUND_TITLE)
    val = tk.StringVar()
    val.set(10)
    epoch = tk.Spinbox(window, from_=10, to=1000, increment=5, textvariable=val, width=5)
    play_btn = tk.Button(window, text='Play Test File', command=lambda: PlaySound(test_path, SND_FILENAME))

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

    infos_menu = InfosMenu(can_menu, text, label, epoch, label_epoch, play_btn, spec, mfcc_choice, spec_choice,
                           ratio, ratio_spinbox, rs, rs_spinbox, label_rs, label_ratio)
    return infos_menu


def init_footer():
    can_footer = tk.Canvas(window, width=WIDTH - WIDTH_BUT, height=LENGTH_BUT, bg=BACKGROUND_TITLE, bd=0,
                           highlightthickness=0, relief='ridge')
    web_button = tk.Button(window, text="Aide", font=("Courrier", 10), bd=0, highlightthickness=0, relief='ridge',
                           command=show_aide)
    foot = Footer(can_footer, web_button)
    return foot


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


def init_sons():
    can = tk.Canvas(window, width=0, height=0, bg=BACKGROUND_SOUND, bd=0,
                    highlightthickness=0, relief='ridge')
    show_audio = tk.Button(window, text="Voir audio", font=("Courrier", 14), fg='black',
                           compound='left', command=show_audio_representation)
    show_spec = tk.Button(window, text="Voir spectrogramme", font=("Courrier", 14), fg='black',
                          compound='left', command=show_spectrogramme)
    show_mfcc = tk.Button(window, text="Voir mfcc", font=("Courrier", 14), fg='black',
                          compound='left', command=show_mfccs)
    show_son = AffichageSon(can, show_audio, show_spec, show_mfcc)
    return show_son


##########################################
# Fonctions internes
##########################################


def leave():
    window.destroy()


def change(new):
    menu_infos.change_percent(new)


def choose_dir_data():
    global data_path
    data_path = filedialog.askdirectory(initialdir="./", title="Selectionnez votre dataset")


def choose_path_csv():
    global path_csv
    path_csv = filedialog.askopenfilename(initialdir="./", title="Selectionnez votre fichier .csv",
                                          filetypes=(("csv  files", "*.csv"), ("all files", "*.*")))


def choose_test_path():
    global test_path
    test_path = filedialog.askopenfilename(initialdir="./", title="Selectionnez votre fichier .wav",
                                           filetypes=(("wav  files", "*.wav"), ("all files", "*.*")))


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


def format_data():
    if data_path == "":
        print("Erreur : Vous devez renseigner le chemin de votre dataset")
        return
    if path_csv == "":
        print("Erreur : Vous devez renseigner le chemin de votre fichier csv")
        return
    if clear_folder("./local_npy_files") == -1:
        print("Erreur : Le dossier ./local_npy_files n'a pas pu être nettoye")
        return
    print("Lancement pour le dossier : " + data_path)
    print("Et le fichier CSV : " + path_csv)
    # if fd.conv_data(data_path, path_csv, ratio=menu_infos.get_ratio(), rs=menu_infos.get_rs(),
    #                 spec=menu_infos.get_spec()) == -1:
    if fd.conv_data(data_path, path_csv, spec=menu_infos.get_spec()) == -1:
        print("Erreur : Un des fichier indiqué n'existe pas")
        return


def run_model():
    global model
    if not (os.path.isfile('./local_npy_files/test_audio.npy')):
        print("Il manque le fichier test_audio.npy essayez de relancer le formatage des fichiers")
        return
    if not (os.path.isfile('./local_npy_files/train_audio.npy')):
        print("Il manque le fichier train_audio.npy essayez de relancer le formatage des fichiers")
        return
    if not (os.path.isfile('./local_npy_files/test_labels.npy')):
        print("Il manque le fichier test_labels.npy essayez de relancer le formatage des fichiers")
        return
    if not (os.path.isfile('./local_npy_files/train_labels.npy')):
        print("Il manque le fichier train_labels.npy essayez de relancer le formatage des fichiers")
        return
    accuracy, model = cnn.run_model(int(menu_infos.get_epochs()))
    menu_infos.change_percent(accuracy)


def predict():
    global model, path_csv
    if test_path == "":
        print("Erreur : Vous devez selectionner un fichier audio .wav en cliquant sur le bouton Test Path")
        return
    if not model:
        accuracy, model = cnn.run_model()
    # pred.create_prediction()
    resultat, le, predicted_proba = pred.print_prediction(test_path, model)
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


def show_spectrogramme():
    audio, sample_rate = librosa.load(test_path, res_type='kaiser_fast')
    spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128,
                                          fmax=11000, power=0.5)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spec, x_axis='time')
    plt.colorbar()
    plt.title('SPECTROGRAMME')
    plt.tight_layout()
    plt.show()


def show_mfccs():
    audio, sample_rate = librosa.load(test_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=100, hop_length=1024, htk=True)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()


def show_audio_representation():
    samplerate, data = read(test_path)
    duration = len(data) / samplerate
    time = np.arange(0, duration, 1 / samplerate)  # time vector

    plt.plot(time, data)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Représentation de l audio')
    plt.show()


##########################################
# Aide
##########################################


class Aide:
    def __init__(self, main_can):
        self.main_can = main_can

    def creation(self):
        f = open("./aide.txt", "r")
        self.main_can.create_text(WIDTH / 2, HEIGHT / 2, font=("Courrier", 12), fill='black', text=f.read())

    def display(self):
        self.main_can.place(x=0, y=0)


def init_aide(win):
    main_can = tk.Canvas(win, width=WIDTH, height=HEIGHT, bd=0,
                         highlightthickness=0, relief='ridge')
    aide = Aide(main_can)
    return aide


def show_aide():
    win = tk.Toplevel(window)
    win.title("Aide Projet L3")
    local_width = WIDTH
    local_height = HEIGHT
    win.geometry(str(local_width) + "x" + str(local_height))
    win.minsize(local_width, local_height)
    win.maxsize(local_width, local_height)
    win.tk.call('wm', 'iconphoto', win, tk.PhotoImage(file="./img/logo.png"))
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
    background_image = tk.PhotoImage(file='./img/background.png')
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
    son.display()

    data_path = ""
    path_csv = ""
    test_path = ""
    model = None

    window.mainloop()
