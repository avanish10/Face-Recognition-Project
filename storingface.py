# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 23:57:34 2019

@author: AVANISH SINGHAL
"""


#Write a python script that captures images from ur webcam video stream'
#Extract all faces from the image frame (using haarcascades)
#Stores the face info in arrays


#Read and show video stream ,Capture images
#Detect faces and show bounding box
#Flatten the largest face image and stor ein numpty array
#Repeat the above for multiple people to generate data


import cv2
import numpy as np

#Initialise the camera

cap = cv2.VideoCapture(0)

#haarcascade file for fAce detection
face_cascade = cv2.CascadeClassifier(r"C:\Users\AVANISH SINGHAL\Desktop\DS coding blocks\machine-learning-online-2018-master\6. Project - Face Recognition\OpenCV Basics\haarcascade_frontalface_alt.xml")
skip = 0
face_data = []
dataset_path = "./dataface/"


file_name = input("Enter the name of  the person: ") 

while True:
    ret,frame = cap.read()
    if False:
        continue
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(frame,1.3,5)#frame,scaling value,no. of neighbours
    faces = sorted(faces,key = lambda f:f[2]*f[3])
    
    
    #Pick the last face because it is the largest face according to area f[2]*f[3]
    
    
    for face in faces[-1:]:
        x,y,w,h = face
        #For bounding box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)#frame,coordinate,wif=dth and height, color,multiple face number
    
    
    #Extract (crop out the required face ):Region of interest
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))
        cv2.imshow("Frame",frame)
        cv2.imshow("Face section",face_section)
    #Store every 10th face
    if(skip%10==0):
        #Store the 10th face    later on
        pass
      
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break


#Convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)


#Save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data successfully saved at"+dataset_path+file_name+'.npy')
cap.release()
cap.destroyAllWindows()