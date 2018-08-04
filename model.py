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
	ID_num = 0
	ID_map = {}
	for imgPath in imagePath:
		faceImg = Image.open(imgPath)
		faceNp = np.array(faceImg,'uint8')
		ID = os.path.split(imgPath)[-1].split('_')[0]
		if ID not in IDs:
			ID_map[ID] = ID_num
			ID_num += 1
		IDs.append(ID)
		faces.append(faceNp)
		cv2.imshow("Training...",faceNp)
		cv2.waitKey(10)
	return IDs,faces,ID_map

IDs,faces,ID_map = getImageswithID(path)
# Training the model
for i in range(len(IDs)):
	IDs[i] = ID_map[IDs[i]]
recog.train(faces,np.array(IDs))
# Saving the model
recog.write('trainingData.yml')
print(list(ID_map.keys()))
np.savetxt("names.txt",np.array(list(ID_map.keys())),fmt = '%s')
np.savetxt("number.txt",np.array(list(ID_map.values())),fmt = '%s')

