# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 11:50:05 2021

@author: mohamed
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import models
import cv2 as cv
from skimage.feature import hog



def prepare_svm(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img=cv.resize(img, (35, 35))
    plt.imshow(img,cmap=plt.cm.binary)
    img = hog(img, block_norm='L1', pixels_per_cell=(8, 8))
    img = np.array(img)
    return img        

def prepare_ann(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img=cv.resize(img, (35, 35))
    plt.imshow(img,cmap=plt.cm.binary)
    img = hog(img, block_norm='L1', pixels_per_cell=(8, 8))
    img = np.array(img)
    return img

import pickle
filename = 'svm_model2.sav'
svm_model = pickle.load(open(filename, 'rb'))
ann_model = models.load_model("ann.model")

CATEGORIES = ["Ahmed Amr","Ali Habib","Mohamed Bebo","Mohamed Labib","Mohamed Mokhtar"]

#GUI---------------------------
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk


class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Iris recognition")
        self.minsize(640, 400)

        self.labelFrame = ttk.LabelFrame(self, text = "Open File")
        self.labelFrame.grid(column = 0, row = 1, padx = 20, pady = 20)

        self.button()


    def button(self):
        self.button = ttk.Button(self.labelFrame, text = "Browse A File",command = self.fileDialog)
        self.button.grid(column = 1, row = 1)


    def fileDialog(self):

        self.filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =
        (("jpeg files","*.jpg"),("all files","*.*")) )
        self.label = ttk.Label(self.labelFrame, text = "")
        self.label.grid(column = 1, row = 2)
        self.label.configure(text = self.filename)
        self.reco_img(self.filename)

    def reco_img(self,img_dir):
        img = cv.imread(img_dir)

        svm_img = prepare_svm(img)
        ann_img = prepare_ann(img)
        
        svm_prediction = svm_model.predict(np.array([svm_img]))
        ann_prediction = ann_model.predict(np.array([ann_img]))
        
        #svm_index=np.argmax(svm_prediction)
        ann_index=np.argmax(ann_prediction)
        print("   ww",svm_prediction)
        r1 = "welcome "+CATEGORIES[ann_index]
        r2 = "welcome "+CATEGORIES[svm_prediction[0]]
        
        self.printOut1(r1)
        self.printOut2(r2)
        
        print("\n\nANN welcome ",CATEGORIES[ann_index])
        print("SVM welcome ",CATEGORIES[svm_prediction[0]])
        
    
    def printOut1(self,text):
        labelFrame = ttk.LabelFrame(self, text = "ANN")
        labelFrame.grid(column = 0, row = 5, padx = 20, pady = 20)
        label = ttk.Label(labelFrame, text = "")
        label.grid(column = 5, row = 5)
        label.master.destroy
        label.configure(text = text)
    
    def printOut2(self,text):
        labelFrame = ttk.LabelFrame(self, text = "SVM")
        labelFrame.grid(column = 0, row = 8, padx = 20, pady = 20)
        label = ttk.Label(labelFrame, text = "")
        label.grid(column = 5, row = 5)
        label.master.destroy
        label.configure(text = text)
root = Root()
root.mainloop()
#--------------------

