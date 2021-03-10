from application.preparation_v2 import format_data as fd, prediction as pred, cnn_model as cnn, find_best_epoch as fbe
from application import generate_csv as gc, get_model as gm
import tkinter as tk
from tkinter import filedialog
import os
import shutil
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read
from keras.models import model_from_json
from pygame import mixer
from pydub import AudioSegment
import sys
from shutil import copyfile


WIDTH = 1600
WIDTH_LINUX = 1500
HEIGHT = 740
HEIGHT_LINUX = 850
TEXT_SIZE_LINUX = 14
TEXT_SIZE_WIN = 12
IMG_RESHAPE_LINUX = 14
IMG_RESHAPE_WIN = 16
BACKGROUND_MENU = '#445976'
BACKGROUND_TITLE = '#2f3d51'
BACKGROUND_SOUND = '#59759c'
LENGTH_BUT = 40
WIDTH_BUT = 200
global data_path, path_csv, test_path, model_path, zip_data, zip_model, model


class Menu:
    """Main menu with all the buttons to manage the program"""
    def __init__(self, can, quit_pic, run_pic, folder_pic, open_but, run_but, quit_but, csv_pic, open_csv_but,
                 format_data_but, format_pic, generate_csv_but, save_csv_img, import_pic, import_model_but,
                 import_data_pic, import_data_but):
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
        self.import_pic = import_pic
        self.import_model_but = import_model_but
        self.import_data_pic = import_data_pic
        self.import_data_but = import_data_but

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
        self.quit_but.config(height=LENGTH_BUT+4, width=WIDTH_BUT-3, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                             relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")
        self.import_model_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                                     relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")
        self.import_data_but.config(height=LENGTH_BUT, width=WIDTH_BUT, bg=BACKGROUND_MENU, bd=1, highlightthickness=0,
                                    relief='ridge', activebackground=BACKGROUND_TITLE, activeforeground="white")

    def display(self):
        self.open_but.place(x=(WIDTH_BUT+4)/2, y=52)
        self.generate_csv_but.place(x=2, y=96)
        self.open_csv_but.place(x=WIDTH_BUT+4, y=96) # 140 / 184 / 228 / 272 / 316
        self.format_data_but.place(x=2, y=140)
        self.import_data_but.place(x=WIDTH_BUT+4, y=140)
        self.run_but.place(x=2, y=184)
        self.import_model_but.place(x=WIDTH_BUT+4, y=184)
        if sys.platform.startswith('linux'):
            self.quit_but.place(x=0, y=HEIGHT_LINUX - 50)
        else:
            self.quit_but.place(x=2, y=HEIGHT - 45)
        self.can.place(x=-1, y=0)


class Header:
    """Small part to display the program icon"""
    def __init__(self, can, icon):
        self.can = can
        self.icon = icon

    def creation(self):
        self.can.create_image(30, 25, image=self.icon)
        self.can.create_text(100, 25, font=("Courrier", 16), fill='white', text="   ML Sound")

    def display(self):
        self.can.place(x=2, y=2)


class InfosMenu:
    """Side Menu who display the informations about the preparation of the model and the saves"""
    def __init__(self, can_menu, text, label, epoch, label_epoch, ratio, ratio_spinbox, rs, rs_spinbox, label_rs,
                 label_ratio, save_model_but, name_model, name_entry, save_data_but, name_data, name_data_entry,
                 best_epoch_but, val, accuracy_label, mfcc, choose_mfcc, choose_spec):
        self.can_menu = can_menu
        self.text = text
        self.label = label
        self.epoch = epoch
        self.label_epoch = label_epoch
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
        self.best_epoch_but = best_epoch_but
        self.val = val
        self.accuracy_label = accuracy_label
        self.mfcc = mfcc
        self.choose_mfcc = choose_mfcc
        self.choose_spec = choose_spec

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

    def define_epochs(self, new_val):
        self.val.set(new_val)

    def get_rs(self):
        return int(self.rs.get())

    def get_ratio(self):
        return float(self.ratio.get())

    def get_save_name(self):
        return self.name_model.get()

    def get_data_name(self):
        return self.name_data.get()

    def get_mfcc(self):
        return self.mfcc.get()

    def display(self):
        if sys.platform.startswith('linux'):
            self.can_menu.place(x=0, y=236)
            self.accuracy_label.place(x=50, y=255)
            self.label.place(x=120, y=250)
            self.label_epoch.place(x=35, y=308)
            self.epoch.place(x=100, y=310)
            self.best_epoch_but.place(x=35, y=335)
            self.choose_mfcc.place(x=35, y=400)
            self.choose_spec.place(x=35, y=425)
            self.label_ratio.place(x=35, y=475)
            self.ratio_spinbox.place(x=100, y=475)
            self.label_rs.place(x=35, y=505)
            self.rs_spinbox.place(x=100, y=505)
            self.name_entry.place(x=30, y=650)
            self.save_model_but.place(x=55, y=680)
            self.save_data_but.place(x=62, y=760)
            self.name_data_entry.place(x=30, y=730)
        else:
            self.can_menu.place(x=2, y=227)
            self.accuracy_label.place(x=40, y=245)
            self.label.place(x=125, y=240)
            self.label_epoch.place(x=40, y=300)
            self.epoch.place(x=110, y=300)
            self.best_epoch_but.place(x=35, y=355)
            self.choose_mfcc.place(x=35, y=400)
            self.choose_spec.place(x=35, y=420)
            self.label_ratio.place(x=35, y=465)
            self.ratio_spinbox.place(x=100, y=465)
            self.label_rs.place(x=35, y=495)
            self.rs_spinbox.place(x=100, y=495)
            self.save_data_but.place(x=58, y=585)
            self.name_data_entry.place(x=20, y=555)
            self.save_model_but.place(x=52, y=660)
            self.name_entry.place(x=20, y=630)


class Footer:
    """Small part who display the version num and the two help buttons"""
    def __init__(self, can, but, help_button):
        self.can = can
        self.but = but
        self.help_button = help_button

    def creation(self):
        self.can.create_text(25, 25, font=("Courrier", 12), fill='white', text="v 0.4")

    def display(self):
        if sys.platform.startswith('linux'):
            self.but.place(x=WIDTH_LINUX - 75, y=HEIGHT_LINUX - 38)
            self.help_button.place(x=WIDTH_LINUX - 150, y=HEIGHT_LINUX - 38)
        else:
            self.but.place(x=WIDTH - 60, y=HEIGHT - 33)
            self.help_button.place(x=WIDTH - 130, y=HEIGHT - 33)
        if sys.platform.startswith('linux'):
            self.can.place(x=WIDTH_BUT + 25, y=HEIGHT_LINUX - LENGTH_BUT - 10)
        else:
            self.can.place(x=WIDTH_BUT, y=HEIGHT - LENGTH_BUT - 2)


class AffichageSon:
    """Display all the informations about the sounds"""
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
        if sys.platform.startswith('linux'):
            self.can.place(x=WIDTH_BUT + 5, y=HEIGHT_LINUX / 2 - 40)
            self.show_audio.place(x=300, y=HEIGHT_LINUX / 2 + 210)
            self.show_spec.place(x=740, y=HEIGHT_LINUX / 2 + 210)
            self.show_mfcc.place(x=1280, y=HEIGHT_LINUX / 2 + 210)
            self.play_btn.place(x=770, y=HEIGHT_LINUX / 2 + 140)
            self.open_test_but.place(x=440, y=HEIGHT_LINUX / 2 + 130)
            self.run_test_but.place(x=1050, y=HEIGHT_LINUX / 2 + 130)
        else:
            self.can.place(x=WIDTH_BUT + 5, y=HEIGHT / 2 - 20)
            self.show_audio.place(x=450, y=HEIGHT / 2 + 140)
            self.show_spec.place(x=800, y=HEIGHT / 2 + 140)
            self.show_mfcc.place(x=1250, y=HEIGHT / 2 + 140)
            self.play_btn.place(x=825, y=HEIGHT / 2 + 80)
            self.open_test_but.place(x=580, y=HEIGHT / 2 + 50)
            self.run_test_but.place(x=1025, y=HEIGHT / 2 + 50)


class AffichageRes:
    """Show the results of the prediction"""
    def __init__(self, can, prediction_label, prediction, results, res_lab):
        self.can = can
        self.prediction_label = prediction_label
        self.prediction = prediction
        self.results = results
        self.res_lab = res_lab

    def new_prediction(self, new):
        self.prediction.set("Le modèle pense qu'il s'agit de : " + new)

    def new_res(self, new):
        for i in range(len(new)):
            if i > 3:
                break
            self.results[i].set(new[i])

    def display(self):
        self.prediction_label.place(x=425, y=HEIGHT / 2 - 300)
        for i in range(len(self.res_lab)):
            self.res_lab[i].place(x=425+i*350, y=HEIGHT / 2 - 250)


class RecapSelect:
    """Recap the choices for the csv path and the data path"""
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
            self.can.place(x=WIDTH_BUT + 25, y=HEIGHT_LINUX / 2 + 315)
            self.data_path_label.place(x=WIDTH_BUT + 35, y=HEIGHT_LINUX / 2 + 320)
            self.csv_path_label.place(x=WIDTH_BUT + 35, y=HEIGHT_LINUX / 2 + 347)
        else:
            self.can.place(x=WIDTH_BUT+2, y=HEIGHT / 2 + 268)
            self.data_path_label.place(x=WIDTH_BUT + 40, y=HEIGHT / 2 + 270)
            self.csv_path_label.place(x=WIDTH_BUT + 40, y=HEIGHT / 2 + 295)


class Console:
    """Print all the errors or the success done by the user"""
    def __init__(self, can, initial_label, variable_label, variable):
        self.can = can
        self.initial_label = initial_label
        self.variable_label = variable_label
        self.variable = variable

    def update_console(self, new):
        self.variable.set("$> " + new)

    def display(self):
        if sys.platform.startswith('linux'):
            self.can.place(x=WIDTH_BUT + 25, y=HEIGHT_LINUX / 2 + 255)
            self.initial_label.place(x=WIDTH_BUT + 40, y=HEIGHT_LINUX / 2 + 265)
            self.variable_label.place(x=WIDTH_BUT + 40, y=HEIGHT_LINUX / 2 + 285)
        else:
            self.can.place(x=WIDTH_BUT, y=HEIGHT / 2 + 208)
            self.initial_label.place(x=WIDTH_BUT + 40, y=HEIGHT / 2 + 218)
            self.variable_label.place(x=WIDTH_BUT + 40, y=HEIGHT / 2 + 238)


def init_window():
    """This function is used to initialize the window, and lock its size"""
    w = tk.Tk()
    w.title("ML Sound")
    if sys.platform.startswith('linux'):
        w.geometry(str(WIDTH_LINUX) + "x" + str(HEIGHT_LINUX))
        w.minsize(WIDTH_LINUX, HEIGHT_LINUX)
        w.maxsize(WIDTH_LINUX, HEIGHT_LINUX)
    else:
        w.geometry(str(WIDTH) + "x" + str(HEIGHT))
        w.minsize(WIDTH, HEIGHT)
        w.maxsize(WIDTH, HEIGHT)
    w.tk.call('wm', 'iconphoto', w, tk.PhotoImage(file="./img/logo.png"))
    return w


def init_menu():
    """This function creates the buttons needed to indicate the paths of the
    different data needed / processing the data"""
    if sys.platform.startswith('linux'):
        size_txt = TEXT_SIZE_LINUX
        size_reshape = IMG_RESHAPE_LINUX
    else:
        size_txt = TEXT_SIZE_WIN
        size_reshape = IMG_RESHAPE_WIN
    folder_img = tk.PhotoImage(file='./img/folder.png').subsample(size_reshape, size_reshape)
    save_csv_img = tk.PhotoImage(file='./img/save_csv.png').subsample(size_reshape, size_reshape)
    run_pic = tk.PhotoImage(file='./img/funnel.png').subsample(size_reshape, size_reshape)
    csv_pic = tk.PhotoImage(file='./img/csv.png').subsample(size_reshape, size_reshape)
    format_pic = tk.PhotoImage(file='./img/format.png').subsample(size_reshape, size_reshape)
    import_data_pic = tk.PhotoImage(file='./img/import_data.png').subsample(size_reshape, size_reshape)
    import_pic = tk.PhotoImage(file='./img/import.png').subsample(size_reshape, size_reshape)
    quit_pic = tk.PhotoImage(file='./img/leave.png').subsample(size_reshape, size_reshape)
    if sys.platform.startswith('linux'):
        can_menu = tk.Canvas(window, width=(WIDTH_BUT+16)*2, height=LENGTH_BUT*5, bg=BACKGROUND_TITLE, bd=0,
                             highlightthickness=0, relief='ridge')
    else:
        can_menu = tk.Canvas(window, width=(WIDTH_BUT+4)*2, height=LENGTH_BUT*5, bg=BACKGROUND_TITLE, bd=0,
                             highlightthickness=0, relief='ridge')
    open_train_but = tk.Button(window, image=folder_img, text="  Data Path", font=("Courrier", size_txt), fg='black',
                               compound='left', command=choose_dir_data)
    generate_csv_but = tk.Button(window, image=save_csv_img, text="  Generate .CSV", font=("Courrier", size_txt), fg='black',
                                 compound='left', command=generate_csv)
    open_csv_but = tk.Button(window, image=csv_pic, text="  CSV Path", font=("Courrier", size_txt), fg='black',
                             compound='left', command=choose_path_csv)
    format_data_but = tk.Button(window, image=format_pic, text="  Format data", font=("Courrier", size_txt), fg='black',
                                compound='left', command=format_data)
    run_train_but = tk.Button(window, image=run_pic, text="  Run Train", font=("Courrier", size_txt), fg='black',
                              compound='left', command=run_model)
    import_data_but = tk.Button(window, image=import_data_pic, text="  Import Data", font=("Courrier", size_txt), fg='black',
                                compound='left', command=import_data)
    import_model_but = tk.Button(window, image=import_pic, text="  Import Model", font=("Courrier", size_txt), fg='black',
                                 compound='left', command=import_model)
    quit_but = tk.Button(window, image=quit_pic, text="  Exit", font=("Courrier", size_txt), fg='black', compound='left',
                         command=leave)
    win_menu = Menu(can_menu, quit_pic, run_pic, folder_img, open_train_but, run_train_but, quit_but, csv_pic,
                    open_csv_but, format_data_but, format_pic, generate_csv_but, save_csv_img, import_pic,
                    import_model_but, import_data_pic, import_data_but)
    return win_menu


def init_header():
    """This is used to initialize the header for the window"""
    icon = tk.PhotoImage(file="./img/logo.png").subsample(12, 12)
    if sys.platform.startswith('linux'):
        can_head = tk.Canvas(window, width=(WIDTH_BUT + 10) *2, height=HEIGHT / 15 + 2, bd=0, highlightthickness=0,
                             relief='ridge', bg=BACKGROUND_TITLE)
    else:
        can_head = tk.Canvas(window, width=(WIDTH_BUT + 2)*2, height=HEIGHT / 15, bd=0, highlightthickness=0,
                             relief='ridge', bg=BACKGROUND_TITLE)
    win_header = Header(can_head, icon)
    return win_header


def init_infos_menu():
    """This is used to indicate all the needed informations to write the text on the buttons"""
    if sys.platform.startswith('linux'):
        can_menu = tk.Canvas(window, width=225, height=570, bg=BACKGROUND_TITLE, bd=0,
                             highlightthickness=0, relief='ridge')
    else:
        can_menu = tk.Canvas(window, width=WIDTH_BUT, height=470, bg=BACKGROUND_TITLE, bd=0,
                             highlightthickness=0, relief='ridge')
    text = tk.StringVar()
    text.set("- %")
    label = tk.Label(window, textvariable=text, font=("Courrier", 14), bg=BACKGROUND_TITLE)
    label_epoch = tk.Label(window, text="Epochs ", font=("Courrier", 10), bg=BACKGROUND_TITLE)
    val = tk.StringVar()
    val.set(10)
    epoch = tk.Spinbox(window, from_=10, to=1000, increment=5, textvariable=val, width=5)
    best_epoch_but = tk.Button(window, text="Find best Epoch", font=("Courrier", 11), fg='black',
                               command=find_best_epoch)
    mfcc = tk.BooleanVar()
    mfcc.set(True)
    choose_spec = tk.Radiobutton(window, text="Spectrogram", variable=mfcc, value=False, bg=BACKGROUND_TITLE)
    choose_mfcc = tk.Radiobutton(window, text="Mfcc", variable=mfcc, value=True, bg=BACKGROUND_TITLE)

    label_ratio = tk.Label(window, text="Ratio ", font=("Courrier", 10), bg=BACKGROUND_TITLE)
    ratio = tk.StringVar()
    ratio.set(0.1)
    ratio_spinbox = tk.Spinbox(window, from_=0, to=0.95, increment=.05, textvariable=ratio, width=5)
    label_rs = tk.Label(window, text="RS ", font=("Courrier", 10), bg=BACKGROUND_TITLE)
    rs = tk.StringVar()
    rs.set(42)
    rs_spinbox = tk.Spinbox(window, from_=0, to=100, increment=1, textvariable=rs, width=5)
    save_data_but = tk.Button(window, text="save data", font=("Courrier", 11), fg='black', command=save_as_data_format)
    name_data = tk.StringVar(value='data')
    name_data_entry = tk.Entry(window, textvariable=name_data)
    save_model_but = tk.Button(window, text="save model", font=("Courrier", 11), fg='black', command=save_as_model)
    name_model = tk.StringVar(value='model')
    name_entry = tk.Entry(window, textvariable=name_model)
    accuracy_label = tk.Label(window, text="Accuracy", font=("Courrier", 10), bg=BACKGROUND_TITLE)
    infos_menu = InfosMenu(can_menu, text, label, epoch, label_epoch, ratio, ratio_spinbox, rs, rs_spinbox, label_rs,
                           label_ratio, save_model_but, name_model, name_entry, save_data_but, name_data,
                           name_data_entry, best_epoch_but, val, accuracy_label, mfcc, choose_mfcc, choose_spec)
    return infos_menu


def init_footer():
    """This is used to initialize the footer for the window"""
    if sys.platform.startswith('linux'):
        can_footer = tk.Canvas(window, width=WIDTH_LINUX - WIDTH_BUT, height=LENGTH_BUT + 15, bg=BACKGROUND_TITLE,
                               bd=0, highlightthickness=0, relief='ridge')
    else:
        can_footer = tk.Canvas(window, width=WIDTH - WIDTH_BUT, height=LENGTH_BUT+4, bg=BACKGROUND_TITLE, bd=0,
                               highlightthickness=0, relief='ridge')
    aide_button = tk.Button(window, text="Aide", font=("Courrier", 10), bd=0, highlightthickness=0, relief='ridge',
                            command=show_aide)

    help_button = tk.Button(window, text="Help", font=("Courrier", 10), bd=0, highlightthickness=0, relief='ridge',
                            command=show_help)

    foot = Footer(can_footer, aide_button, help_button)
    return foot


def init_resultats():
    """This is used to return the results of the prediction"""
    if sys.platform.startswith('linux'):
        can = tk.Canvas(window, width=WIDTH_LINUX - WIDTH_BUT, height=HEIGHT / 2 - 50, bg=BACKGROUND_SOUND, bd=0,
                        highlightthickness=0, relief='ridge')
    else:
        can = tk.Canvas(window, width=WIDTH - WIDTH_BUT, height=HEIGHT / 2 - 50, bg=BACKGROUND_SOUND, bd=0,
                        highlightthickness=0, relief='ridge')
    prediction = tk.StringVar()
    prediction.set("")
    prediction_label = tk.Label(window, textvariable=prediction, font=("Courrier", 16), bg=BACKGROUND_SOUND)

    results = []
    res_lab = []

    for i in range(3):
        res_curr = tk.StringVar()
        res_curr.set("")
        results_label = tk.Label(window, textvariable=res_curr, font=("Courrier", 14), bg=BACKGROUND_SOUND)
        results.append(res_curr)
        res_lab.append(results_label)

    result = AffichageRes(can, prediction_label, prediction, results, res_lab)
    return result


def init_sons():
    """This is used to create the buttons used to see the graphic representations of a sound"""
    wav_pic = tk.PhotoImage(file='./img/wav.png').subsample(14, 14)
    process_pic = tk.PhotoImage(file='./img/process.png').subsample(14, 14)
    can = tk.Canvas(window, width=0, height=0, bg=BACKGROUND_SOUND, bd=0,
                    highlightthickness=0, relief='ridge')
    show_audio = tk.Button(window, text="Show audio", font=("Courrier", 14), fg='black',
                           compound='left', command=show_audio_representation)
    show_spec = tk.Button(window, text="Show spectrogram", font=("Courrier", 14), fg='black',
                          compound='left', command=show_spectrogramme)
    show_mfcc = tk.Button(window, text="Show mfcc", font=("Courrier", 14), fg='black',
                          compound='left', command=show_mfccs)
    play_btn = tk.Button(window, text='Play Test File', font=("Courrier", 14), command=lambda: run_test_audio())
    open_test_but = tk.Button(window, image=wav_pic, text="  Test Path", font=("Courrier", 14), fg='black',
                              compound='left', command=choose_test_path)
    run_test_but = tk.Button(window, image=process_pic, text="  Run Test", font=("Courrier", 14), fg='black',
                             compound='left', command=predict)
    show_son = AffichageSon(can, show_audio, show_spec, show_mfcc, play_btn, wav_pic, process_pic, open_test_but,
                            run_test_but)
    return show_son


def init_model():
    """This is used to initialize the model using model.json and the obtained accuracy"""
    global model
    try:
        # load json and create model
        file = open("./local_saves/model/model.json", 'r')
        file_acc = open("./local_saves/accuracy.txt", 'r')
    except IOError:
        cons.update_console("Error : File not accessible")
        return None
    model_json = file.read()
    acc_str = file_acc.read()
    if acc_str != '':
        accuracy = int(float(acc_str))
        menu_infos.change_percent(accuracy)
        model_local = model_from_json(model_json)
        file.close()
        # load weights
        model_local.load_weights("./local_saves/model/model.h5")
        cons.update_console("Model load")
        model = model_local


def init_recap_selec():
    """Initialise the show data path and csv path section"""
    if sys.platform.startswith('linux'):
        can = tk.Canvas(window, width=WIDTH_LINUX - WIDTH_BUT - 20, height=60, bg="white", bd=0, highlightthickness=0,
                        relief='ridge')
    else:
        can = tk.Canvas(window, width=WIDTH - WIDTH_BUT, height=60, bg="white", bd=0, highlightthickness=0,
                        relief='ridge')
    data_path_var = tk.StringVar()
    data_path_var.set("Data path :")
    csv_path_var = tk.StringVar()
    csv_path_var.set("CSV path :")
    data_path_label = tk.Label(window, textvariable=data_path_var, font=("Courrier", 11))
    csv_path_label = tk.Label(window, textvariable=csv_path_var, font=("Courrier", 11))
    recapit = RecapSelect(can, data_path_label, data_path_var, csv_path_label, csv_path_var)
    return recapit


def init_console():
    if sys.platform.startswith('linux'):
        can = tk.Canvas(window, width=WIDTH_LINUX - WIDTH_BUT - 20, height=60, bg="black", bd=0, highlightthickness=0,
                        relief='ridge')
    else:
        can = tk.Canvas(window, width=WIDTH - WIDTH_BUT, height=60, bg="black", bd=0, highlightthickness=0,
                        relief='ridge')
    variable = tk.StringVar()
    variable.set("$> ")
    initial_label = tk.Label(window, text="Console", font=("Courrier", 11), fg='white', bg='black')
    variable_label = tk.Label(window, textvariable=variable, font=("Courrier", 11), fg='white', bg='black')
    cons = Console(can, initial_label, variable_label, variable)
    return cons


def run_test_audio():
    """This is used to play the sound so you can hear it. If it's an mp3, it will use pygame to play it,
        otherwise, it will use playsound"""
    if test_path != '':
        # if test_path.endswith('.mp3'):
        mixer.init()
        mixer.music.load(test_path)
        mixer.music.play()
        cons.update_console("Play the sound : " + test_path)
        # else:
        #     playsound(test_path)


def leave(event=None):
    """This is used to quit the application"""
    cons.update_console("^c")
    window.destroy()


def change(new):
    """This allows the estimated accuracy to be changed using the value in parameters"""
    menu_infos.change_percent(new)


def choose_dir_data():
    """This is used to indicate the path to the dataset"""
    global data_path
    new = filedialog.askdirectory(initialdir="./", title="Selectionnez votre dataset")
    if new != '':
        data_path = new
        recap.update_data_path(data_path)
        cons.update_console("data path update : " + data_path)


def choose_path_csv():
    """This is used to indicate the path to the .csv"""
    global path_csv
    new = filedialog.askopenfilename(initialdir="./", title="Selectionnez votre fichier .csv",
                                     filetypes=(("csv  files", "*.csv"), ("all files", "*.*")))
    if new != '':
        path_csv = new
        recap.update_csv_path(path_csv)
        cons.update_console("csv path update : " + path_csv)


def choose_test_path():
    """This is used to indicate the path to the tested file"""
    global test_path
    new = filedialog.askopenfilename(initialdir="./", title="Selectionnez votre fichier son",
                                     filetypes=[(".mp3", ".mp3"), (".wav", ".wav"), ("all files", ".*")])
    if new != '':
        test_path = new


def clear_folder(folder):
    """This is used to clear the paths to the different"""
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            cons.update_console('Failed to delete %s. Reason: %s' % (file_path, e))
            return -1
    return 0


def copy_floder_content(origin, dest):
    """Copy all the content of a repository (origin) to another one (dest)
    :param origin: path of the original repository
    :param dest: path of the destination repository
    :return: 0 if everything ok -1 if error"""
    for filename in os.listdir(origin):
        file_path = os.path.join(origin, filename)
        try:
            copyfile(file_path, dest)
        except Exception as e:
            cons.update_console("Failed to copy " + str(file_path) + " Reason : " + str(e))
            return -1
    return 0


def format_data():
    """This is used to format the data using the different paths indicated"""
    global menu_infos
    if data_path == "":
        cons.update_console("Error : You need to specify th path to your data")
        return
    if path_csv == "":
        cons.update_console("Error : You need to specify th path to your csv file")
        return
    if not os.path.exists('./local_saves/data_format'):
        os.mkdir('./local_saves/data_format')
    elif clear_folder("./local_saves/data_format") == -1:
        cons.update_console("Error : Le directory ./local_saves could not be cleaned")
        return
    get_rs = menu_infos.get_rs()
    get_ratio = menu_infos.get_ratio()
    get_mfcc = menu_infos.get_mfcc()
    cons.update_console("Start with data : " + data_path + " / rs : " + str(get_rs) + " / ratio : " + str(get_ratio))
    fd.get_the_data(data_path, path_csv, "./local_saves/data_format/class_label.txt", get_ratio, get_rs, get_mfcc)


def run_model():
    """This is used to run the model using cnn_model.py after formatting the data"""
    global model
    if not (os.path.isfile('./local_saves/data_format/test_audio_mfcc.npy')):
        cons.update_console("The file test_audio is missing try to format again")
        return
    if not (os.path.isfile('./local_saves/data_format/train_audio_mfcc.npy')):
        cons.update_console("The file train_audio is missing try to format again")
        return
    if not (os.path.isfile('./local_saves/data_format/test_labels_mfcc.npy')):
        cons.update_console("The file test_labels is missing try to format again")
        return
    if not (os.path.isfile('./local_saves/data_format/train_labels_mfcc.npy')):
        cons.update_console("The file train_labels is missing try to format again")
        return
    accuracy, model = cnn.run_model(int(menu_infos.get_epochs()), menu_infos.get_mfcc())
    menu_infos.change_percent(accuracy)
    cons.update_console("Run finish, accuracy of : " + str(accuracy))


def predict():
    """This is used to get the prediction of what bird / bat species the test sound corresponds to, and then print it"""
    global model, path_csv, menu_infos
    if test_path == "":
        cons.update_console(
            "Error : You have to select an audio file .wav or .mp3 by clicking on the 'test path' button")
        return
    if not model:
        accuracy, model = cnn.run_model(int(menu_infos.get_epochs()), menu_infos.get_mfcc())
    get_mfcc = menu_infos.get_mfcc()
    resultat, le, predicted_proba = pred.print_prediction(test_path, model, get_mfcc)
    res.new_prediction(resultat)
    list_prob, cmpt, curr = [], 0, 0
    all_prob = ""
    for i in range(len(predicted_proba)):
        category = le.inverse_transform(np.array([i]))
        current_classe = category[0]
        if type(current_classe) is tuple:
            current_classe = str(current_classe[0])
        prob = int(float(predicted_proba[i]) * 100)
        if cmpt%10 == 0 and cmpt != 0 and cmpt < 30:
            list_prob.append(all_prob)
            all_prob = ""
            all_prob = current_classe + " : " + str(prob) + " %\n"
        else:
            if (cmpt+1)%10 == 0 and cmpt < 20:
                all_prob += current_classe + " : " + str(prob)
            else:
                all_prob += current_classe + " : " + str(prob) + " %\n"
        cmpt = cmpt + 1
    list_prob.append(all_prob)
    res.new_res(list_prob)


def show_spectrogramme():
    """This is used to show the spectrogram corresponding to the sound that is being tested"""
    if test_path == "":
        cons.update_console(
            "Error : You have to select an audio file .wav or .mp3 by clicking on the 'test path' button")
        return
    if not (os.path.isfile(test_path)):
        cons.update_console(
            "Error : The file selected doesn't exists select an audio file .wav or .mp3 by clicking on the "
            "'test path' button")
        return
    audio, sample_rate = librosa.load(test_path, res_type='kaiser_fast')
    spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=50,
                                          fmax=11000, power=0.5)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spec, x_axis='time')
    plt.colorbar()
    plt.title('SPECTROGRAMME')
    plt.tight_layout()
    plt.show()


def show_mfccs():
    """This is used to show the mfcc corresponding to the sound that is being tested"""
    if test_path == "":
        cons.update_console(
            "Error : You have to select an audio file .wav or .mp3 by clicking on the 'test path' button")
        return
    if not (os.path.isfile(test_path)):
        cons.update_console(
            "Error : The file selected doesn't exists select an audio file .wav or .mp3 by clicking on the "
            "'test path' button")
        return
    audio, sample_rate = librosa.load(test_path, res_type='kaiser_fast')
    m = np.mean(audio)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=50, hop_length=1024, htk=True)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()


def show_audio_representation():
    """This is used to show the audio representation corresponding to the sound that is being tested"""
    if test_path == "":
        cons.update_console(
            "Error : You have to select an audio file .wav or .mp3 by clicking on the 'test path' button")
        return
    if not (os.path.isfile(test_path)):
        cons.update_console(
            "Error : The file selected doesn't exists select an audio file .wav or .mp3 by clicking on the "
            "'test path' button")
        return
    if test_path.endswith('.mp3'):
        sound = AudioSegment.from_mp3(test_path)
        dst = './local_saves/current.wav'
        sound.export(dst, format="wav")
    else:
        dst = test_path
    try:
        samplerate, data = read(dst)
        duration = len(data) / samplerate
        time = np.arange(0, duration, 1 / samplerate)  # time vector
        plt.plot(time, data)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('Audio representation')
        plt.show()
    except ValueError:
        cons.update_console("Error : x and y must have same first dimension")


def save_as_model():
    """This is used save the current model in the "local save" folder to be able to use it later without
    re-creating every step"""
    if not os.path.isdir("./local_saves/model"):
        return
    if os.listdir('./local_saves/model') == 0:
        return
    cons.update_console("Save as ...")
    save_model_path = filedialog.askdirectory(initialdir="./",
                                              title="Choose a path to save your model")
    val_name = menu_infos.get_save_name()
    if val_name == '':
        shutil.make_archive(save_model_path + "/my_model", "zip", "./local_saves/model")
    else:
        shutil.make_archive(save_model_path + "/" + val_name, "zip", "./local_saves/model")
    cons.update_console("Model saved")


def save_as_data_format():
    """This is used to save the data for later"""
    if not os.path.isdir("./local_saves/data_format"):
        cons.update_console("Error : Try to reformat the data")
        return
    if os.listdir('./local_saves/data_format') == 0:
        cons.update_console("Error : Try to reformat the data")
        return
    cons.update_console("Save as ...")
    save_model_path = filedialog.askdirectory(initialdir="./",
                                              title="Choose a path to save your data")
    val_name = menu_infos.get_data_name()
    if val_name == '':
        shutil.make_archive(save_model_path + "/my_data", "zip", "./local_saves/data_format")
    else:
        shutil.make_archive(save_model_path + "/" + val_name, "zip", "./local_saves/data_format")
    cons.update_console("Data saved")


def generate_csv():
    """This is used to generate a .csv (Spreadsheet) using the Data set indicated"""
    global data_path, path_csv
    cons.update_console("Generate csv")
    if data_path == "":
        cons.update_console("Error : You have to enter your data path")
        return
    if gc.generate(data_path):
        path_csv = './local_saves/auto_generate.csv'
        recap.update_csv_path(path_csv)
        cons.update_console("Csv generated and path updated")


def set_zip_data(win):
    """Get the data from a zip file"""
    win.destroy()
    global model_path, model
    zip_path = filedialog.askopenfilename(initialdir="./", title="Selectionnez votre fichier .zip",
                                          filetypes=(("zip  files", "*.zip"), ("all files", "*.*")))
    if zip_path != '':
        clear_folder("./local_saves/data_format")
        os.mkdir("./local_unzip")
        tmp_path = "./local_unzip"
        gm.local_unzip(zip_path, tmp_path)
        copy_floder_content("./local_unzip", "./local_saves/data_format")
        shutil.rmtree(tmp_path)
        cons.update_console("Data Load From : " + zip_path)


def set_rep_data(win):
    """Get the data from a repository"""
    win.destroy()
    rep_path = filedialog.askdirectory(initialdir="./", title="Selectionnez votre dataset")
    if rep_path != '':
        clear_folder("./local_saves/data_format")
        copy_floder_content(rep_path, "./local_saves/data_format")
        cons.update_console("Data Load From : " + rep_path)


def import_data():
    """Import the data by asking the user"""
    win = tk.Toplevel(window)
    win.title("Zip or Rep ?")
    win.geometry(str(150) + "x" + str(75))
    win.minsize(150, 75)
    win.maxsize(150, 75)
    win.tk.call('wm', 'iconphoto', win, tk.PhotoImage(file="./img/logo.png"))
    zip_but = tk.Button(win, text="ZIP", font=("Courrier", 14), fg='black', compound='left',
                        command=lambda: set_zip_data(win))
    rep_but = tk.Button(win, text="REP", font=("Courrier", 14), fg='black', compound='left',
                        command=lambda: set_rep_data(win))
    zip_but.pack(side='left')
    rep_but.pack(side='right')


def set_zip(win):
    """Get the model from a zip file"""
    global model_path, model
    win.destroy()
    zip_path = filedialog.askopenfilename(initialdir="./", title="Selectionnez votre fichier .zip",
                                          filetypes=(("zip  files", "*.zip"), ("all files", "*.*")))
    if zip_path != '':
        model_path = zip_path
        model = gm.get_model(model_path)
        change(-1)
        cons.update_console("Model Load From : " + zip_path)


def set_rep(win):
    """Get the model from a repository"""
    global model_path, model
    win.destroy()
    rep_path = filedialog.askdirectory(initialdir="./", title="Selectionnez votre dataset")
    if rep_path != '':
        model_path = rep_path
        model = gm.get_model(model_path)
        change(-1)
        cons.update_console("Model Load From : " + rep_path)


def import_model():
    """Import a model by asking the user """
    win = tk.Toplevel(window)
    win.title("Zip or Rep ?")
    win.geometry(str(150) + "x" + str(75))
    win.minsize(150, 75)
    win.maxsize(150, 75)
    win.tk.call('wm', 'iconphoto', win, tk.PhotoImage(file="./img/logo.png"))
    zip_but = tk.Button(win, text="ZIP", font=("Courrier", 14), fg='black', compound='left',
                        command=lambda: set_zip(win))
    rep_but = tk.Button(win, text="REP", font=("Courrier", 14), fg='black', compound='left',
                        command=lambda: set_rep(win))
    zip_but.pack(side='left')
    rep_but.pack(side='right')


def find_best_epoch():
    """Call the function, aims at find best epochs"""
    best = fbe.get_best(menu_infos.get_mfcc())
    menu_infos.define_epochs(best)
    console_msg = "New best epoch find : " + str(best)
    cons.update_console(console_msg)


class Aide:
    """
    This class is used to print the help / instructions should the user press the help button in the bottom right corner
    """
    def __init__(self, main_can):
        self.main_can = main_can

    def creation(self, path_help):
        f = open(path_help, "r")
        if sys.platform.startswith('linux'):
            self.main_can.create_text(WIDTH_LINUX / 2, HEIGHT_LINUX / 2, font=("Courrier", 10), fill='black',
                                      text=f.read())
        else:
            self.main_can.create_text((WIDTH+100) / 2, HEIGHT / 2, font=("Courrier", 10), fill='black', text=f.read())

    def display(self):
        self.main_can.place(x=0, y=0)


def init_aide(win):
    """Is used to create the help window"""
    if sys.platform.startswith('linux'):
        main_can = tk.Canvas(win, width=WIDTH_LINUX, height=HEIGHT_LINUX, bd=0, highlightthickness=0, relief='ridge')
    else:
        main_can = tk.Canvas(win, width=WIDTH, height=HEIGHT+100, bd=0, highlightthickness=0, relief='ridge')
    aide = Aide(main_can)
    return aide


def create_help(path_help):
    """Aims to create the help section in english"""
    win = tk.Toplevel(window)
    win.title("Help ML Sound")
    if sys.platform.startswith('linux'):
        local_width = WIDTH_LINUX
        local_height = HEIGHT_LINUX
    else:
        local_width = WIDTH
        local_height = HEIGHT+100
    win.geometry(str(local_width) + "x" + str(local_height))
    win.minsize(local_width, local_height)
    win.maxsize(local_width, local_height)
    win.tk.call('wm', 'iconphoto', win, tk.PhotoImage(file="./img/logo.png"))
    aide = init_aide(win)
    aide.creation(path_help)
    aide.display()


def show_aide():
    """Display the french version of the help"""
    create_help("./resources/aide.txt")


def show_help():
    """Display the english version of the help"""
    create_help("./resources/aide_en.txt")


def load_model():
    """Used to load a pre-existing model into the application"""
    full_path = "./model.zip"
    return gm.get_model(full_path)


def start():
    global window, menu, header, menu_infos, footer, res, son, recap, cons
    # Creation of the window
    window = init_window()

    # Initialisation of the application background
    if sys.platform.startswith('linux'):
        background_image = tk.PhotoImage(file='./img/background_linux.png')
    else:
        background_image = tk.PhotoImage(file='./img/background.png')
    background_label = tk.Label(window, image=background_image)
    background_label.place(x=0, y=0)

    # Creation of the menu
    menu = init_menu()
    menu.config()
    menu.display()

    # Creation of the header
    header = init_header()
    header.creation()
    header.display()

    # Creation of the options place
    menu_infos = init_infos_menu()
    menu_infos.display()

    # Creation of the footer
    footer = init_footer()
    footer.creation()
    footer.display()

    # Creation of the results display part
    res = init_resultats()
    res.display()

    # Creation of the sound display part
    son = init_sons()
    son.config()
    son.display()

    # creation of the parts to recap informations for the user
    recap = init_recap_selec()
    recap.display()

    cons = init_console()
    cons.display()

    # Initialisation des variables
    # data_path = ""
    # path_csv = ""
    # test_path = ""
    # model_path = ""
    # zip_model = False
    init_model()

    window.bind_all('<Control-q>', leave)

    window.mainloop()