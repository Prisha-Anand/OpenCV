import os
import numpy as np
import cv2 as cv
from datetime import datetime
def markAttendance(label1):
    with open('class_atttd.csv','r+') as f:
        dat=f.readlines()
        nl=[]
        for l in dat:
            ent=l.split(',')
            nl.append(ent[0])
        if label1 not in nl:
            now=datetime.now()
            d=now.strftime("%H:%M:%S")
            f.writelines(f'\n{label1},{d}')

haar_cascade = cv.CascadeClassifier('cascade.xml')

people = ['alex','boseman','prisha','trump']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img2 = cv.VideoCapture(0)
(_, img)=img2.read()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
grays=[]
cv.imshow('Person', gray)


faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    faces_roi = gray[y:y+h,x:x+w]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')
    cv.putText(img,people[label],(x,y),cv.FONT_HERSHEY_PLAIN,1.5,(0,255,0),2)
    markAttendance(people[label])
cv.imshow('Detected Face', img)

cv.waitKey(0)