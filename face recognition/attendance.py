from base64 import encode
from email.mime import image
from re import M
from sre_constants import SUCCESS
import cv2
import numpy as np
import face_recognition
import os



path = "imageattendance"
images = []
classname = []
myList  = os.listdir(path)


for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    name = os.path.splitext(cl)[0]
    classname.append(name)

def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


encodelistknown = findEncodings(images)
print ("encoding completed")


cap = cv2.VideoCapture(0)

while True:
    success , img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodingCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)


    for encodeFace,faceLoc in zip(encodingCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodelistknown,encodeFace)
        facedis = face_recognition.face_distance(encodelistknown,encodeFace)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]:
            name = classname[matchIndex].upper()
            print (name)
            y1,x2,y2,x1 = faceLoc


            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4


            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    cv2.imshow("webcam",img)
    cv2.waitKey(1)






# face_loc = face_recognition.face_locations(imgelon)[0]
# encodeelon = face_recognition.face_encodings(imgelon)[0]
# cv2.rectangle(imgelon,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(255,0,0),2)

# faceLocTest = face_recognition.face_locations(imgtest)[0]
# encodeTest = face_recognition.face_encodings(imgtest)[0]
# cv2.rectangle(imgtest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
# # print(face_loc)
 
# result = face_recognition.compare_faces([encodeelon],encodeTest)
# face_dis = face_recognition.face_distance([encodeelon],encodeTest)