#Import the required Libraries
from PIL import Image
import os
import numpy as np
import cv2
#----------------------------------------------

#Initialize the Face Recognition model, prebuilt in the cv2
recog = cv2.face.LBPHFaceRecognizer_create()
path = "Dataset"

# A function to create the training dataset
def getImageswithID(path):
	imagePath = [os.path.join(path,f) for f in os.listdir(path)]
	faces = []
	IDs = []

	for imgPath in imagePath:
		faceImg = Image.open(imgPath)
		faceNp = np.array(faceImg,'uint8')
		ID = int(os.path.split(imgPath)[-1].split('_')[0])
		IDs.append(ID)
		faces.append(faceNp)
		cv2.imshow("Training...",faceNp)
		cv2.waitKey(10)
	return IDs,faces

IDs,faces = getImageswithID(path)
# Training the model
recog.train(faces,np.array(IDs))
# Saving the model
recog.write('trainingData.yml')

