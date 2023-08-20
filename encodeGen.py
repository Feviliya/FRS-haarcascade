import cv2
import face_recognition
import pickle
import os

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://facerecogrealtime-bb854-default-rtdb.firebaseio.com/",
    'storageBucket':"facerecogrealtime-bb854.appspot.com"
})


folderPath = 'images'
pathList = os.listdir(folderPath)
# print(pathList)
imgList=[]
victimIds=[]
print("Uploading images to bucket....")
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    victimIds.append(os.path.splitext(path)[0])

    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)
print("Upload completed !!")

def findEncodings(imagesList):
    encodeList=[]
    for img in imagesList:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
print("Encoding started ...")
encodeListKnown= findEncodings(imgList)
encodeListKnownWithIds=[encodeListKnown,victimIds]
print("Encoding completed")

file=open("EncodeFile.p",'wb')
pickle.dump(encodeListKnownWithIds,file)
file.close()
print("File saved")