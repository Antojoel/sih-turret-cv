from multiprocessing.connection import wait
from operator import imod
import cv2
import numpy as np
import face_recognition

imgelon = face_recognition.load_image_file('ImagesBasic/elonmusk.jpg')
imgelon = cv2.cvtColor(imgelon, cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('ImagesBasic/elonmusk2.jpg')
imgtest = cv2.cvtColor(imgtest, cv2.COLOR_BGR2RGB)

face_loc = face_recognition.face_locations(imgelon)[0]
encodeelon = face_recognition.face_encodings(imgelon)[0]
cv2.rectangle(imgelon,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(255,0,0),2)

faceLocTest = face_recognition.face_locations(imgtest)[0]
encodeTest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
# print(face_loc)
 
result = face_recognition.compare_faces([encodeelon],encodeTest)
face_dis = face_recognition.face_distance([encodeelon],encodeTest)
print (result,face_dis)


# cv2.imshow('elonmusk', imgelon)
# cv2.imshow('elontest', imgtest)
cv2.waitKey(0)