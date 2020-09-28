#!/usr/bin/env python
# coding: utf-8

#this code is adapted from:
#https://github.com/gauravtheP/Real-Time-Facial-Expression-Recognition/blob/master/RealTimeFacialExpressionRecognition/Real_Time_Prediction.ipynb


import  cv2
import numpy as np
from keras.models import load_model


EMOTION_DICT = {1:"AFRAID", 2:"ANGRY", 3:"DISGUST", 4:"HAPPY", 5:"NEUTRAL", 6:"SAD", 7:"SURPRISE"}

vgg = load_model("vgg-frontal-crop.hdf5")


def return_prediction(path):
    #converting image to gray scale and save it
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path, gray)
    
    #detect face in image, crop it then resize it then save it
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        face_clip = img[y:y+h, x:x+w]
        cv2.imwrite(path, cv2.resize(face_clip, (300, 300)))
    
    #read the processed image then make prediction and display the result
    read_image = cv2.imread(path)
    read_image = read_image.reshape(1, read_image.shape[0], read_image.shape[1], read_image.shape[2])
    
    #read_image_final = read_image/255.0
    read_image_final = read_image
    
    top_pred = vgg.predict(read_image_final)
    emotion_label = top_pred[0].argmax() + 1
    return EMOTION_DICT[emotion_label]



def rerun(text, cap):
    while(True):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Emotion detected: "+str(text), (95,30), font, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.putText(img, "press space to predict", (5,470), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    
        cv2.putText(img, "hold q to quit", (460,470), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
        
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) == ord(' '):
            cv2.imwrite("test.jpg", img)
            text = return_prediction("test.jpg")
            first_run(text, cap)
            break
            
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

cap = cv2.VideoCapture(0)

def first_run(text, cap):
    while(True):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Emotion detected: "+str(text), (95,30), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.putText(img, "press space to predict", (5,470), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
        cv2.putText(img, "hold q to quit", (460,470), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) == ord(' '):
            cv2.imwrite("test.jpg", img)
            text = return_prediction("test.jpg")
            rerun(text, cap)
            break
            
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


first_run("None", cap)