from PIL import Image
import os
import numpy as np
import cv2
#----------------------------------------------

recog = cv2.face.LBPHFaceRecognizer_create()
path = "Dataset"

def getImageswithID(path):
	imagePath = [os.path.join(path,f) for f in os.listdir(path)]
	faces = []
	IDs = []

	for imgPath in imagePath:
		faceImg = Image.open(imgPath)
		faceNp = np.array(faceImg,'uint8')
		ID = int(os.path.split(imgPath)[-1].split('_')[0])
		if ID == 10:
                        faces.append(faceNp)
		IDs.append(ID)
		cv2.imshow("Training...",faceNp)
		cv2.waitKey(10)
	return IDs,faces
IDs,faces = getImageswithID(path)
recog.train(faces,np.array(IDs))
recog.write('trainingData.yml')

