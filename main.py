import cv2
import numpy as np
import face_recognition

image=face_recognition.load_image_file('Images/input5.jpeg')
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

face_locations=face_recognition.face_locations(image)

print(face_locations)

for i in range(len(face_locations)):

    image=cv2.rectangle(image,(face_locations[i][3],face_locations[i][0]),(face_locations[i][1],face_locations[i][2]),(255,0,255),2)

cv2.imshow('Group of people',image)
cv2.waitKey(0)