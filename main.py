import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt
import cv2
from easygui import *
import os
from PIL import Image, ImageTk
from itertools import count
import tkinter as tk
import string
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import keras
from pygame import mixer  # Load the popular external library
from gtts import gTTS


def func1():
    r = sr.Recognizer()

    arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
           'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        i = 0
        while True:
            print("Recording....")
            audio = r.listen(source)
            # recognize speech using Sphinx
            try:
                a = r.recognize_google(audio)
                a = a.lower()
                print('You Said: ' + a.lower())

                for c in string.punctuation:
                    a = a.replace(c, "")

                if(a.lower() == 'goodbye' or a.lower() == 'good bye' or a.lower() == 'bye'):
                    print("oops!Time To say good bye")
                    break
                else:
                    for i in range(len(a)):
                        if(a[i] in arr):

                            ImageAddress = 'letters/'+a[i]+'.jpg'
                            ImageItself = Image.open(ImageAddress)
                            ImageNumpyFormat = np.asarray(ImageItself)
                            plt.imshow(ImageNumpyFormat)
                            plt.draw()
                            plt.pause(0.8)
                        else:
                            continue

            except:
                print(" ")
            plt.close()


def func2():
    model = keras.models.load_model("asl_classifier.h5")
    labels_dict = {0: '0',
                   1: 'A',
                   2: 'B',
                   3: 'C',
                   4: 'D',
                   5: 'E',
                   6: 'F',
                   7: 'G',
                   8: 'H',
                   9: 'I',
                   10: 'J',
                   11: 'K',
                   12: 'L',
                   13: 'M',
                   14: 'N',
                   15: 'O',
                   16: 'P',
                   17: "Q",
                   18: 'R',
                   19: 'S',
                   20: 'T',
                   21: 'U',
                   22: 'V',
                   23: 'W',
                   24: 'X',
                   25: 'Y',
                   26: 'Z'}
    color_dict = (0, 255, 0)
    x = 0
    y = 0
    w = 64
    h = 64
    img_size = 128
    minValue = 70
    source = cv2.VideoCapture(0)
    count = 0
    string = " "
    prev = " "
    prev_val = 0
    while(True):
        ret, img = source.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.rectangle(img,(x,y),(x+w,y+h),color_dict,2)
        cv2.rectangle(img, (24, 24), (250, 250), color_dict, 2)
        crop_img = gray[24:250, 24:250]
        count = count + 1
        if(count % 100 == 0):
            prev_val = count
        cv2.putText(img, str(prev_val//100), (300, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        blur = cv2.GaussianBlur(crop_img, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, res = cv2.threshold(
            th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        resized = cv2.resize(res, (img_size, img_size))
        normalized = resized/255.0
        reshaped = np.reshape(normalized, (1, img_size, img_size, 1))
        result = model.predict(reshaped)
        # print(result)
        label = np.argmax(result, axis=1)[0]
        if(count == 300):
            count = 99
            prev = labels_dict[label]
            if(label == 0):
                string = string + " "
                # if(len(string)==1 or string[len(string)] != " "):

            else:
                string = string + prev

        cv2.putText(img, prev, (24, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        cv2.putText(img, string, (275, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.imshow("Gray", res)
        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)

        if(key == 27):  # press Esc. to exit
            break
    print(string)
    cv2.destroyAllWindows()
    source.release()

    cv2.destroyAllWindows()

    # This module is imported so that we can
    # play the converted audio

    # The text that you want to convert to audio

    # Language in which you want to convert
    language = 'en'
    # Passing the text and language to the engine,
    # here we have marked slow=False. Which tells
    # the module that the converted audio should
    # have a high speed
    if string == ' ':
        return
    myobj = gTTS(text=string, lang=language, slow=False)

    # Saving the converted audio in a mp3 file named
    # welcome
    myobj.save("welcome.mp3")

    # Playing the converted file
    mixer.init()
    mixer.music.load('welcome.mp3')
    mixer.music.play()


while 1:
    image = "signlang.png"
    msg = "Two way Sign Language Translator"
    choices = ["Audio to Sign", "Quit", "Sign to Audio"]
    reply = buttonbox(msg, image=image, choices=choices)
    if reply == choices[0]:
        func1()
    if reply == choices[1]:
        quit()
    if reply == choices[2]:
        func2()
