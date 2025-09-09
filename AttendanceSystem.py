import sys
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path='ImageAttendance'
images=[]
classNames=[]
mylist=os.listdir(path)
print(mylist)
for cl in mylist:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodeings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
        except IndexError as e:
            print(e)
            sys.exit(1)

        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtstring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')





encodeListknown=findEncodeings(images)
print('Encoding complete')

#cap=cv2.VideoCapture(0)
#cap=face_recognition.load_image_file('Images/input.jpeg')
cap=cv2.imread('Images/input5.jpeg')
while True:
    #success,img= cap.read()
    #imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)

    facecurFrame = face_recognition.face_locations(cap)
    encodecurFrame = face_recognition.face_encodings(cap,facecurFrame)

    for encodeFace,faceLoc in zip(encodecurFrame,facecurFrame):
        matches=face_recognition.compare_faces(encodeListknown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListknown,encodeFace)
        #print(faceDis)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            print(name)
            markAttendance(name)

    #cv2.imshow('Webcam',imgs)
    #cv2.waitKey(1)

