# -*- coding: utf-8 -*-
"""
Created on Wed 23 Oct 2020

@author: Pierre Barbat Maximilien Cetre Thomas Corcoral
"""

##########################################
#############  Importation  ##############
##########################################

import webbrowser
import random as rd
from tkinter import filedialog
from tkinter import *
from shutil import copyfile
import math
from deeplearn import *

##########################################
##########  Variables globales  ##########
##########################################

WIDTH = 1500
HEIGHT = 800
AUTHORS = "BARBAT PIERRE, CETRE MAXIMILIEN, CORCORAL THOMAS"

##########################################
#######  Définition des classes  #########
##########################################


class Menu:
    def __init__(self, can, quit_pic, loupe_pic, folder_pic, open_but, run_but, quit_but):
        self.can = can
        self.quit_pic = quit_pic
        self.loupe_pic = loupe_pic
        self.folder_pic = folder_pic
        self.open_but = open_but
        self.run_but = run_but
        self.quit_but = quit_but

    def config(self):
        self.run_but.config(height=110, width=175)
        self.open_but.config(height=110, width=175)
        self.quit_but.config(height=110, width=175)

    def display(self):
        self.quit_but.place(x=2, y=675)
        self.open_but.place(x=2, y=275)
        self.run_but.place(x=2, y=400)
        self.can.place(x=-1, y=HEIGHT / 10)


class Footer:
    def __init__(self, can, but):
        self.can = can
        self.but = but

    def creation(self):
        self.can.create_text(240, 28, font=("Courrier", 12), fill='white', text=AUTHORS)

    def display(self):
        self.but.place(x=WIDTH-125, y=HEIGHT-42)
        self.can.place(x=205, y=HEIGHT - HEIGHT / 15)


class Header:
    def __init__(self, can, icon):
        self.can = can
        self.icon = icon

    def creation(self):
        self.can.create_image(40, 40, image=self.icon)
        self.can.create_text(250, 20, font=("Courrier", 22), fill='white', text="Projet DeepLearning")
        self.can.create_text(513, 55, font=("Courrier", 16), fill='white',
                           text="Bienvenue dans notre application de reconnaissance par réseau de neurone")
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
            self.picture = self.picture.subsample(self.w_resize-1)
        else:
            self.picture = self.picture.subsample(self.h_resize)
        self.can.delete("CUR")
        self.can.create_image(0, 0, anchor=NW, image=self.picture, tags="CUR")

    def display(self):
        if self.h_resize < self.w_resize:
            self.local_h_resize = (600-self.picture.height()-1)/2
            self.local_w_resize = (600-self.picture.width()-1)/2
        else:
            self.local_h_resize = (600-self.picture.height())/2
            self.local_w_resize = (600-self.picture.width())/2
        self.can.place(x=250+self.local_w_resize, y=HEIGHT/10+35+self.local_h_resize)

##########################################
###########  Fonctions  ##################
##########################################


def open_website():
    webbrowser.open_new("https://projet.xnh.fr/")


def leave(window):
    window.destroy()


def init_window():
    w = Tk()
    w.title("DeepLearning")
    w.geometry(str(WIDTH) + "x" + str(HEIGHT))
    w.minsize(WIDTH, HEIGHT)
    w.maxsize(WIDTH, HEIGHT)
    return w


def init_menu(window):
    folder_img = PhotoImage(file='venv/assets/img/folder.png').subsample(6, 6)
    loupe_pic = PhotoImage(file='venv/assets/img/loupe.png').subsample(6, 6)
    quit_pic = PhotoImage(file='venv/assets/img/remove.png').subsample(6, 6)

    can_menu = Canvas(window, width=205, height=HEIGHT - HEIGHT / 10, bg="grey")
    open_but = Button(window, image=folder_img, text="OPEN", font=("Courrier", 21), fg='white', compound='bottom',
                      command=lambda: ouvrir(curr_pic))
    run_but = Button(window, image=loupe_pic, text="START", font=("Courrier", 21), fg='white',
                     compound='bottom', command=testdeep)
    quit_but = Button(window, image=quit_pic, text="QUIT", font=("Courrier", 21), fg='white',
                      compound='bottom',
                      command=lambda: leave(window))

    menu = Menu(can_menu, quit_pic, loupe_pic, folder_img, open_but, run_but, quit_but)
    return menu


def init_header(window):
    icon = PhotoImage(file="venv/assets/img/logo.png").subsample(8, 8)
    can_head = Canvas(window, width=WIDTH, height=HEIGHT / 10, bg="#5F6060")
    header = Header(can_head, icon)
    return header


def init_footer(window):
    can_footer = Canvas(window, width=WIDTH - 205, height=HEIGHT / 15, bg="grey")
    web_button = Button(window, text="Site internet", font=("Courrier", 10), command=open_website)
    footer = Footer(can_footer, web_button)
    return footer


def init_current(window):
    current = Canvas(window, width=600, height=600)
    default = PhotoImage(file="venv/assets/img/defaut.png")
    curr_pic = Current_pic(current, default)
    return curr_pic


def ouvrir(curr_pic):
    filename = filedialog.askopenfilename(initialdir="/", title="Choisissez votre image",
                                          filetypes=(("png files", "*.png"), ("all files", "*.*")))
    print(filename)
    r = rd.randint(0, 9999999)
    dst = "venv/assets/upload/" + str(r) + ".png"
    copyfile(filename, dst)
    curr_pic.update(dst)
    curr_pic.display()

##########################################
###########  lancement  ##################
##########################################


if __name__ == "__main__":
    # Création de la fenêtre
    window = init_window()

    #Création de la photo courante
    curr_pic = init_current(window)
    curr_pic.display()

    # Création du menu
    menu = init_menu(window)
    menu.config()
    menu.display()

    # Création du Header
    header = init_header(window)
    header.creation()
    header.display()

    # Création du footer
    footer = init_footer(window)
    footer.creation()
    footer.display()

    window.mainloop()
