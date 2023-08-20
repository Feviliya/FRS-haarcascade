import pickle
import numpy as np
import cv2,time
import face_recognition
import cvzone

import geocoder

from datetime  import datetime
import argparse
import os

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://facerecogrealtime-bb854-default-rtdb.firebaseio.com/",
    'storageBucket':"facerecogrealtime-bb854.appspot.com"
})

bucket=storage.bucket()
print("Starting webcam.....")

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

imgBackground=cv2.imread('resources/background.png')

#load encoding file
file=open("EncodeFile.p",'rb')
encodeListKnownWithIds=pickle.load(file)
file.close()
encodeListKnown,victimIds=encodeListKnownWithIds
print(victimIds)

counter = 0
vId = 0
imgVictim = []

def get_geolocation():
    g = geocoder.ip('me')
    if g.latlng and g.city:
        latitude, longitude = g.latlng
        city = g.city
        return latitude, longitude, city
    else:
        return None, None, None



while(True):
    success,img =cap.read()
    imgResized=cv2.resize(img,(0,0),None,0.25,0.25)
    imgResized=cv2.cvtColor(imgResized,cv2.COLOR_BGR2RGB)

    faceCurFrame=face_recognition.face_locations(imgResized)
    encodeCurFrame = face_recognition.face_encodings(imgResized,faceCurFrame)

    imgBackground[120:120+480,111:111+640]=img
    
    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        # print("matches",matches)
        # print("Face Dis: ",faceDis)
        matchIndex = np.argmin(faceDis)
        # print("The matched index is: ",matchIndex)
        if matches[matchIndex]:
            print("Match found: ",victimIds[matchIndex])
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            bbox = 111+ x1,120+y1,x2-x1,y2-y1
            imgBackground = cvzone.cornerRect(imgBackground,bbox,rt=0)
            vId = victimIds[matchIndex]
            if counter==0:
                latitude, longitude, city = get_geolocation()
                if latitude is not None and longitude is not None and city is not None:
                    print(f"Latitude: {latitude}, Longitude: {longitude}, Location: {city}")
                else:
                    print("Unable to retrieve geolocation.")
                counter=1
                frame=img
                if frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
                    for x, y, w, h in faces:
                        capImg = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        exact_time = datetime.now().strftime('%Y-%b-%d-%H-%S-%f')
                        cv2.imwrite("face detected" + str(exact_time) + ".jpg", capImg)
    
    if counter != 0:
        victimInfo = db.reference(f'Victims/{vId}').get()
        print(victimInfo)
        blob = bucket.get_blob(f'images/{vId}.jpg')
        array = np.frombuffer(blob.download_as_string(),np.uint8)
        imgVictim = cv2.imdecode(array,cv2.COLOR_BGRA2BGR)
        cv2.putText(imgBackground, str( victimInfo['name']), (bbox[0], bbox[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 209, 28), 2, cv2.LINE_AA)

        cv2.putText(imgBackground, str(victimInfo['age']), (861, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)  # Fill the area with black
        cv2.putText(imgBackground, str(victimInfo['age']), (861, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (202, 250, 252), 2, cv2.LINE_AA)

        cv2.putText(imgBackground, str(victimInfo['sex']), (861, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)  # Fill the area with black
        cv2.putText(imgBackground, str(victimInfo['sex']), (861, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (202, 250, 252), 2, cv2.LINE_AA)

        cv2.putText(imgBackground, str(victimInfo['city']), (861, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)  # Fill the area with black
        cv2.putText(imgBackground, str(victimInfo['city']), (861, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (202, 250, 252), 2, cv2.LINE_AA)

        counter += 1

    cv2.imshow("template", imgBackground)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()