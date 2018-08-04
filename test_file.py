import cv2
import numpy as np


facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('trainingData.yml')

id = 0
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0,0,255)

# Add names of the people to this map
id_map = np.loadtxt('names.txt',dtype = str)
print(id_map)


cam = cv2.VideoCapture(0)

while True:
	ret,img = cam.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = facedetect.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		id,conf = rec.predict(cv2.resize(gray[y:y+h,x:x+w],(96,96)))
		if (conf <= 100):
			cv2.putText(img,str(id_map[id])+"  "+str(conf),(x,y+h),fontFace,fontScale,fontColor)
		else:
			cv2.putText(img,"Unknown",(x,y+h),fontFace,fontScale,fontColor)
	cv2.imshow('face',img)
	if(cv2.waitKey(1) == 27):
		break
cam.release()
cv2.destroyAllWindows()
