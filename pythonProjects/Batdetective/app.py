# -*- coding: utf-8 -*-
"""
Created on Wed 23 Oct 2020
@author: Pierre Barbat Maximilien Cetre Thomas Corcoral
"""

##########################################
# Importation
##########################################

import webbrowser
# from Tkinter import filedialog
import tkFileDialog
from Tkinter import *
import math
from tkFont import Font
import os
import csv
import shutil
import run_detector

# from run_detector import running_detector

##########################################
# Variables globales
##########################################

WIDTH = 1500
HEIGHT = 800
AUTHORS = "BARBAT PIERRE, CETRE MAXIMILIEN, CORCORAL THOMAS"


##########################################
# Définition des classes
##########################################


class Menu:
    def __init__(self, can, quit_pic, loupe_pic, folder_pic, open_but, run_but, quit_but, precision, pr,
                 data, first_choice, second_choice, third_choice, model, first_model, second_model, check_time,
                 time_exp):
        self.can = can
        self.quit_pic = quit_pic
        self.loupe_pic = loupe_pic
        self.folder_pic = folder_pic
        self.open_but = open_but
        self.run_but = run_but
        self.quit_but = quit_but
        self.precision = precision
        self.pr = pr
        self.tab = []
        self.data = data
        self.first_choice = first_choice
        self.second_choice = second_choice
        self.third_choice = third_choice
        self.model = model
        self.first_model = first_model
        self.second_model = second_model
        self.time_exp = time_exp
        self.check_time = check_time

    def get_data_choice(self):
        return self.data.get()

    def get_model_choice(self):
        return self.model.get()

    def get_time_choice(self):
        return self.time_exp.get()

    def add_local(self, tab):
        for el in self.tab:
            el.destroy()
        self.tab = tab

    def config(self):
        self.run_but.config(height=110, width=175)
        self.open_but.config(height=110, width=175)
        self.quit_but.config(height=110, width=175)

    def display(self):
        self.quit_but.place(x=2, y=675)
        self.open_but.place(x=2, y=275)
        self.run_but.place(x=2, y=400)
        self.can.place(x=-1, y=HEIGHT / 10)
        self.precision.place(x=55, y=180)
        self.pr.place(x=30, y=205)
        self.first_choice.place(x=40, y=535)
        self.second_choice.place(x=40, y=555)
        self.third_choice.place(x=40, y=575)
        self.first_model.place(x=40, y=625)
        self.second_model.place(x=40, y=645)
        self.check_time.place(x=40, y=140)

    def get_value(self):
        res = self.pr.get()
        resint = 0.95
        if len(res) == 0:
            return resint
        try:
            resint = float(res)
        except ValueError:
            return resint
        if resint < 0:
            return 0
        if resint > 100:
            return 1
        return resint / 100


class Footer:
    def __init__(self, can, but):
        self.can = can
        self.but = but

    def creation(self):
        self.can.create_text(240, 28, font=("Courrier", 12), fill='white', text=AUTHORS)

    def display(self):
        self.but.place(x=WIDTH - 125, y=HEIGHT - 42)
        self.can.place(x=205, y=HEIGHT - HEIGHT / 15)


class Header:
    def __init__(self, can, icon):
        self.can = can
        self.icon = icon

    def creation(self):
        self.can.create_image(40, 40, image=self.icon)
        self.can.create_text(250, 20, font=("Courrier", 22), fill='white', text="Projet DeepLearning")
        self.can.create_text(325, 55, font=("Courrier", 16), fill='white',
                             text="Détection du signal acoustique des chauves-souris")

    def display(self):
        self.can.place(x=-1, y=-1)


class Current_pic:
    def __init__(self, can, picture):
        self.can = can
        self.picture = picture
        self.size = 600
        self.h_resize = int(math.ceil(self.picture.height() / self.size))
        self.w_resize = int(math.ceil(self.picture.width() / self.size))
        self.can.create_image(0, 0, anchor=NW, image=self.picture, tags="CUR")
        self.local_h_resize = 0
        self.local_w_resize = 0

    def update(self, path):
        self.picture = PhotoImage(file=path)
        self.h_resize = int(math.ceil(self.picture.height() / self.size))
        self.w_resize = int(math.ceil(self.picture.width() / self.size))
        print(self.picture.height())
        if self.h_resize < self.w_resize:
            self.picture = self.picture.subsample(self.w_resize - 1)
        else:
            self.picture = self.picture.subsample(self.h_resize)
        self.can.delete("CUR")
        self.can.create_image(0, 0, anchor=NW, image=self.picture, tags="CUR")

    def display(self):
        if self.h_resize < self.w_resize:
            self.local_h_resize = (600 - self.picture.height() - 1) / 2
            self.local_w_resize = (600 - self.picture.width() - 1) / 2
        else:
            self.local_h_resize = (600 - self.picture.height()) / 2
            self.local_w_resize = (600 - self.picture.width()) / 2
        self.can.place(x=250 + self.local_w_resize, y=HEIGHT / 10 + 35 + self.local_h_resize)


##########################################
# Fonctions
##########################################


def open_website():
    webbrowser.open_new("https://projet.xnh.fr/")


def leave():
    window.destroy()


def init_window():
    w = Tk()
    w.title("DeepLearning")
    w.geometry(str(WIDTH) + "x" + str(HEIGHT))
    w.minsize(WIDTH, HEIGHT)
    w.maxsize(WIDTH, HEIGHT)
    return w


def run_detect():
    with open('./results/op_file.csv', 'w') as csvfile:
        fieldnames = ['file_name', 'detection_time', 'detection_prob']
        csv.DictWriter(csvfile, fieldnames=fieldnames)
    reload(run_detector)
    run_detector.running_detector(menu.get_value(), menu.get_data_choice(), menu.get_model_choice(),
                                  menu.get_time_choice())
    f = open('./results/op_file.csv')
    csv_f = csv.reader(f)
    y = 220
    x_dif = 0
    cmpt = 0
    arr = []
    for row in csv_f:
        if cmpt == 15:
            x_dif += 280
            y = 220
            cmpt = 0
        local_time = Label(window, text=row[1])
        local_prob = Label(window, text=row[2])
        local_time.place(x=370 + x_dif, y=y)
        local_prob.place(x=520 + x_dif, y=y)
        y += 30
        arr.append(local_time)
        arr.append(local_prob)
        cmpt += 1
    menu.add_local(arr)


def init_menu():
    folder_img = PhotoImage(file='./img/folder.png').subsample(6, 6)
    loupe_pic = PhotoImage(file='./img/loupe.png').subsample(6, 6)
    quit_pic = PhotoImage(file='./img/remove.png').subsample(6, 6)

    can_menu = Canvas(window, width=205, height=HEIGHT - HEIGHT / 10, bg="grey")
    open_but = Button(window, image=folder_img, text="OPEN", font=Font(family="Helvetica", size=21), fg='white',
                      compound='bottom',
                      command=lambda: ouvrir())
    run_but = Button(window, image=loupe_pic, text="START", font=Font(family="Helvetica", size=21), fg='white',
                     compound='bottom', command=lambda: run_detect())
    quit_but = Button(window, image=quit_pic, text="QUIT", font=Font(family="Helvetica", size=21), fg='white',
                      compound='bottom',
                      command=lambda: leave())

    precision = Label(window, text="Precision en %")

    time_exp = IntVar()
    check_time = Checkbutton(window, text="Time expansion", variable=time_exp)

    data = IntVar()
    first_choice = Radiobutton(window, text="Bulgaria", variable=data, value=1)
    first_choice.select()
    second_choice = Radiobutton(window, text="Norfolk", variable=data, value=2)
    third_choice = Radiobutton(window, text="UK", variable=data, value=3)

    model = IntVar()
    first_model = Radiobutton(window, text="192K", variable=model, value=1)
    first_model.select()
    second_model = Radiobutton(window, text="Normal", variable=model, value=2)

    pr = Entry(window)

    m = Menu(can_menu, quit_pic, loupe_pic, folder_img, open_but, run_but, quit_but, precision, pr,
             data, first_choice, second_choice, third_choice, model, first_model, second_model, check_time,
             time_exp)
    return m


def init_header():
    icon = PhotoImage(file="./img/logo.png").subsample(8, 8)
    can_head = Canvas(window, width=WIDTH, height=HEIGHT / 10, bg="#5F6060")
    h = Header(can_head, icon)
    return h


def init_footer():
    can_footer = Canvas(window, width=WIDTH - 205, height=HEIGHT / 15, bg="grey")
    web_button = Button(window, text="Site internet", font=("Courrier", 10), command=open_website)
    f = Footer(can_footer, web_button)
    return f


def init_current():
    current = Canvas(window, width=600, height=600)
    default = PhotoImage(file="./img/defaut.png")
    c = Current_pic(current, default)
    return c


def ouvrir():
    filename = tkFileDialog.askopenfilename(initialdir="/", title="Choisissez votre image",
                                            filetypes=(("wav files", "*.wav"), ("all files", "*.*")))

    if filename:
        print(filename)

        # On retire tous les fichiers du dossier d'analyse pour n'avoir qu'un seul fichier à annalyser
        for root, dirs, files in os.walk('./wavs'):
            for i in files:
                os.remove(os.path.join(root, i))

        # r = rd.randint(0, 9999999)
        # dst = "./wavs/" + str(r) + ".wav"
        shutil.copy(filename, './wavs/')
        # copyfile(filename, dst)
        # curr_pic.update(dst)
        # curr_pic.display()


##########################################
# lancement
##########################################


if __name__ == "__main__":
    # Création de la fenêtre
    window = init_window()

    # Création de la photo courante
    #curr_pic = init_current()
    #curr_pic.display()

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

    window.mainloop()
