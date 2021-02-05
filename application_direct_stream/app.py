# -*- coding: utf-8 -*-
"""
Created on Mon 01 Feb 2021

@author: Pierre Barbat Maximilien Cetre Thomas Corcoral

The purpose of this file is to create a window to analyse the direct input of
the microphone of the computer
"""

##########################################
# Imports
##########################################

import tkinter as tk
import sys
from tkinter import filedialog
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np
from math import *
import matplotlib.pyplot as plt
import pyaudio
import new_audio_process as nap
import give_prediction_arr as gpa


##########################################
# Globals variables
##########################################

WIDTH = 1300
WIDTH_LINUX = 1500
HEIGHT = 740
HEIGHT_LINUX = 850
BACKGROUND_MENU = '#445976'
BACKGROUND_TITLE = '#2f3d51'
BACKGROUND_SOUND = '#59759c'
LENGTH_BUT = 40
WIDTH_BUT = 175
RATE = 22500
CHUNK = 1024 * 2
NMFCC_MFCC = 50
global path_h5
global path_json
global model
global path_labels

##########################################
# Classes definition
##########################################

"""
This part is for the appearance of all the element inside the window, like the size of it, where the buttons are
For instance, once the accuracy percentage reaches a certain threshold, it will change color to indicate ifyou should 
trust or not the prediction
"""


class Menu:
    def __init__(self, can_menu, h5_img, json_img, txt_pic, start_pic, quit_pic, open_h5_but, open_json_but,
                 open_txt_but, start_but, quit_but):
        self.can_menu = can_menu
        self.h5_img = h5_img
        self.json_img = json_img
        self.txt_pic = txt_pic
        self.start_pic = start_pic
        self.quit_pic = quit_pic
        self.open_h5_but = open_h5_but
        self.open_json_but = open_json_but
        self.start_but = start_but
        self.quit_but = quit_but
        self.open_txt_but = open_txt_but

    def config(self):
        self.open_h5_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                            relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")
        self.open_json_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                                     relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")
        self.start_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                             relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")
        self.open_txt_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                              relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")
        self.quit_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                             relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")

    def display(self):
        self.open_h5_but.place(x=2, y=52)
        self.open_json_but.place(x=2, y=96)
        self.start_but.place(x=2, y=140)
        self.open_txt_but.place(x=2, y=184)
        if sys.platform.startswith('linux'):
            self.quit_but.place(x=0, y=HEIGHT_LINUX - 50)
        else:
            self.quit_but.place(x=2, y=HEIGHT - 45)
        self.can_menu.place(x=-1, y=0)


class Header:
    def __init__(self, can, icon):
        self.can = can
        self.icon = icon

    def creation(self):
        self.can.create_image(30, 25, image=self.icon)
        self.can.create_text(100, 25, font=("Courrier", 16), fill='white', text="   L3 project")

    def display(self):
        self.can.place(x=2, y=2)


class Footer:
    def __init__(self, can):
        self.can = can

    def creation(self):
        self.can.create_text(25, 25, font=("Courrier", 12), fill='white', text="v 0.3")

    def display(self):
        if sys.platform.startswith('linux'):
            self.can.place(x=WIDTH_BUT + 25, y=HEIGHT_LINUX - LENGTH_BUT - 10)
        else:
            self.can.place(x=WIDTH_BUT + 5, y=HEIGHT - LENGTH_BUT - 2)


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
        if sys.platform.startswith('linux'):
            self.can.place(x=WIDTH_BUT + 50, y=HEIGHT_LINUX / 2 + 300)
            self.data_path_label.place(x=WIDTH_BUT + 60, y=HEIGHT_LINUX / 2 + 305)
            self.csv_path_label.place(x=WIDTH_BUT + 60, y=HEIGHT_LINUX / 2 + 332)
        else:
            self.can.place(x=WIDTH_BUT + 20, y=HEIGHT / 2 + 265)
            self.data_path_label.place(x=WIDTH_BUT + 40, y=HEIGHT / 2 + 270)
            self.csv_path_label.place(x=WIDTH_BUT + 40, y=HEIGHT / 2 + 295)


class ShowLivePrediction:
    def __init__(self, can):
        self.can = can
        self.rects = []
        self.labs = []

    def get_can(self):
        return self.can

    def config_rect(self, rects):
        self.rects = rects

    def config_labs(self,labs):
        self.labs = labs

    def update_bar(self, values):
        for i in range(len(values)):
            x0, y0, x1, y1 = self.can.coords(self.rects[i])
            x1 = x1 + values[i] * 300
            print(x0, (x1+values[i]*300))
            self.can.coords(self.rects[i], x0, y0, x1, y1)

    def display(self):
        self.can.place(x=WIDTH_BUT + 20, y=LENGTH_BUT)
        current_h = LENGTH_BUT + LENGTH_BUT / 3
        current_w = WIDTH_BUT + 50
        for i in range(len(self.labs)):
            # print(self.labs[i])
            if i % 13 == 0 and i != 0:
                current_w = current_w + 500
                current_h = LENGTH_BUT
            self.labs[i].place(x=current_w, y=current_h)
            current_h = current_h + 40


##########################################
# Graphics functions
##########################################


# This function is used to initialize the window, and lock its size
def init_window():
    w = tk.Tk()
    w.title("L3 project")
    if sys.platform.startswith('linux'):
        w.geometry(str(WIDTH_LINUX) + "x" + str(HEIGHT_LINUX))
        w.minsize(WIDTH_LINUX, HEIGHT_LINUX)
        w.maxsize(WIDTH_LINUX, HEIGHT_LINUX)
    else:
        w.geometry(str(WIDTH) + "x" + str(HEIGHT))
        w.minsize(WIDTH, HEIGHT)
        w.maxsize(WIDTH, HEIGHT)
    w.tk.call('wm', 'iconphoto', w, tk.PhotoImage(file="../img/logo.png"))
    return w


# This function creates the buttons needed to indicate the paths of the different data needed / processing the data
def init_menu():
    h5_img = tk.PhotoImage(file='../img/logo_h5.png').subsample(14, 14)
    json_img = tk.PhotoImage(file='../img/json.png').subsample(14, 14)
    start_pic = tk.PhotoImage(file='../img/process.png').subsample(14, 14)
    txt_pic = tk.PhotoImage(file='../img/txt.png').subsample(14, 14)
    quit_pic = tk.PhotoImage(file='../img/leave.png').subsample(14, 14)
    can_menu = tk.Canvas(window, width=0, height=0)
    open_h5_but = tk.Button(window, image=h5_img, text="  .h5 Path", font=("Courrier", 14), fg='black',
                               compound='left', command=choose_h5_path)
    open_json_but = tk.Button(window, image=json_img, text="  .JSON Path", font=("Courrier", 14), fg='black',
                                 compound='left', command=choose_json_path)
    open_txt_but = tk.Button(window, image=txt_pic, text="  Labels Path", font=("Courrier", 14), fg='black',
                          compound='left', command=choose_label_txt)
    start_but = tk.Button(window, image=start_pic, text="  Start Stream", font=("Courrier", 14), fg='black',
                             compound='left', command=run_stream)
    quit_but = tk.Button(window, image=quit_pic, text="  Exit", font=("Courrier", 14), fg='black', compound='left',
                         command=leave)
    win_menu = Menu(can_menu, h5_img, json_img, txt_pic, start_pic, quit_pic, open_h5_but, open_json_but, start_but,
                    open_txt_but, quit_but)
    return win_menu


# This is used to initialize the header for the window
def init_header():
    icon = tk.PhotoImage(file="../img/logo.png").subsample(12, 12)
    if sys.platform.startswith('linux'):
        can_head = tk.Canvas(window, width=WIDTH_BUT + 26, height=HEIGHT / 15 + 2, bd=0, highlightthickness=0,
                             relief='ridge', bg=BACKGROUND_TITLE)
    else:
        can_head = tk.Canvas(window, width=WIDTH_BUT + 4, height=HEIGHT / 15, bd=0, highlightthickness=0,
                             relief='ridge', bg=BACKGROUND_TITLE)
    win_header = Header(can_head, icon)
    return win_header


# This is used to initialize the footer for the window
def init_footer():
    if sys.platform.startswith('linux'):
        can_footer = tk.Canvas(window, width=WIDTH_LINUX - WIDTH_BUT, height=LENGTH_BUT+15, bg=BACKGROUND_TITLE,
                               bd=0, highlightthickness=0, relief='ridge')
    else:
        can_footer = tk.Canvas(window, width=WIDTH - WIDTH_BUT, height=LENGTH_BUT, bg=BACKGROUND_TITLE, bd=0,
                               highlightthickness=0, relief='ridge')

    foot = Footer(can_footer)
    return foot


def init_recap_selec():
    if sys.platform.startswith('linux'):
        can = tk.Canvas(window, width=WIDTH_LINUX - WIDTH_BUT - 80, height=60, bg="white", bd=0, highlightthickness=0,
                        relief='ridge')
    else:
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


def init_live_predictions():
    if sys.platform.startswith('linux'):
        can = tk.Canvas(window, width=WIDTH_LINUX - WIDTH_BUT - 80, height=HEIGHT_LINUX - LENGTH_BUT * 2 - 80,
                        bg="white", bd=0, highlightthickness=0, relief='ridge')
    else:
        can = tk.Canvas(window, width=WIDTH - WIDTH_BUT - 40, height=HEIGHT - LENGTH_BUT * 2 - 100, bg="white",
                        bd=0, highlightthickness=0, relief='ridge')
    show_pred = ShowLivePrediction(can)
    return show_pred

##########################################
# Intern functions
##########################################


# This is used to quit the application
def leave():
    window.destroy()


def choose_h5_path():
    global path_h5
    new = filedialog.askopenfilename(initialdir="./", title="Selectionnez votre fichier .h5",
                                     filetypes=(("h5  files", "*.h5"), ("all files", "*.*")))
    if new != '':
        path_h5 = new
        recap.update_csv_path(path_h5)


def choose_json_path():
    global path_json
    new = filedialog.askopenfilename(initialdir="./", title="Selectionnez votre fichier .json",
                                     filetypes=(("json  files", "*.json"), ("all files", "*.*")))
    if new != '':
        path_json = new
        recap.update_csv_path(path_json)


def choose_label_txt():
    global path_labels
    new = filedialog.askopenfilename(initialdir="./", title="Selectionnez votre fichier .txt",
                                     filetypes=(("txt  files", "*.txt"), ("all files", "*.*")))
    if new != '':
        path_labels = new
        recap.update_csv_path(path_json)


def read_labels():
    global path_labels
    class_label = []
    with open(path_labels, 'r') as filehandle:
        for line in filehandle:
            current_place = line[:-1]
            class_label.append(current_place)
    return class_label


# Each bar as a maximum size of 300 ! (x1 - x0 <= 300)
def create_progress_bar(classes):
    height_bar = 30
    current_h = LENGTH_BUT
    current_w = 200
    current_can = show_pred.get_can()
    rects = []
    labs = []
    for i in range(len(classes)):
        if i%13 == 0 and i != 0:
            current_w = current_w + 500
            current_h = LENGTH_BUT
        current_rect = current_can.create_rectangle(current_w, current_h, current_w, current_h - height_bar, fill="orange")
        current_label = tk.Label(window, text=classes[i], font=("Courrier", 14), bg='WHITE')
        rects.append(current_rect)
        labs.append(current_label)
        current_h = current_h + 40
    show_pred.config_rect(rects)
    show_pred.config_labs(labs)
    show_pred.display()


def prepare_screen():
    global path_labels
    if path_labels != '':
        class_label = read_labels()
        class_label = list(dict.fromkeys(class_label))
        le = LabelEncoder()
        to_categorical(le.fit_transform(class_label))
        the_classes = list(le.classes_)
        # print(the_classes)
        # print(len(the_classes))
        create_progress_bar(the_classes)
        return class_label
    else:
        print("Error : You need to specify your labels text file")


def run_stream():
    global model
    class_label = prepare_screen()
    if path_json != '' and path_h5 != '':
        try:
            file = open(path_json, 'r')
            model_json = file.read()
            model = model_from_json(model_json)
            file.close()
            model.load_weights(path_h5)
            # print("RUN")
            stream_listener(class_label)
        except:
            print("Error : Verify your files !")


##########################################
# Stream Listener
##########################################


def soundanalyse(stream, a_stream):
    data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
    res = np.concatenate((data, a_stream), axis=None)
    return res


def stream_listener(class_label):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    count = 0
    audio_stream = np.empty(0)
    values = []
    while count < 30:
        print(count)
        audio_stream = soundanalyse(stream, audio_stream)
        count = count + 1
    prep_audio = nap.process_the_audio(audio_stream)
    prep_audio = np.array(prep_audio)
    values = gpa.give_pred(model, prep_audio, class_label, False)
    show_pred.update_bar(values)
    show_pred.display()
    stream.stop_stream()
    stream.close()
    p.terminate()


##########################################
# Main
##########################################


if __name__ == "__main__":
    # Création de la fenêtre
    window = init_window()

    # Initialisation du fond d'écran de l'application
    if sys.platform.startswith('linux'):
        background_image = tk.PhotoImage(file='../img/background_linux.png')
    else:
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

    # Création du footer
    footer = init_footer()
    footer.creation()
    footer.display()

    # Création des informations concernant les chemins
    recap = init_recap_selec()
    recap.display()

    show_pred = init_live_predictions()
    show_pred.display()

    path_labels = ''
    path_json = ''
    path_h5 = ''

    window.mainloop()
