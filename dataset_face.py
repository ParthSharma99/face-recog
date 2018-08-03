import numpy as np
import cv2

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

sampleNum = 0
uid = input('enter user id')

cam = cv2.VideoCapture(0)

while True:
	ret,img = cam.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	for sampleNum in range(100):
                cv2.imwrite('Dataset/' + str(uid) + "_" + str(sampleNum) + '.jpg',cv2.resize(gray , (96,96)))
##
##	faces = facedetect.detectMultiScale(gray,1.3,5)
##
##	for (x,y,w,h) in faces:
##		cv2.imwrite('Dataset/' + str(uid) + "_" + str(sampleNum) + '.jpg',cv2.resize(gray[ y:y+h, x :x+w ] , (96,96)))
##		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
##		cv2.waitKey(100)
##		sampleNum += 1
	cv2.imshow("face",img)
	cv2.waitKey(1)
	if (sampleNum >= 100):
		break
cam.release()
cv2.destroyAllWindows()
