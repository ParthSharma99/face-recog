import cv2
import urllib.request
import ssl
context = ssl._create_unverified_context()
url = "https://192.168.1.2:8080/video"

##face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
##right_eye_detect = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
##left_eye_detect = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
hand_detect = cv2.CascadeClassifier(r'C:\Users\parth\Desktop\HandControl\haarcascades\palm.xml')
#cam = cv2.VideoCapture(0)

while True:
    imageUrl = urllib.request.urlopen(url,context = context).read()
    imageNp = np.array(bytearray(imgUrl),dtype = np.uint8)
    img = cv2.imdecode(imageNp,-1)
    cv2.imshow("face",img) 
##    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hands = hand_detect.detectMultiScale(gray, 1.3, 5)
##    eql = cv2.equalizeHist(img)
##    faces = face_detect.detectMultiScale(gray, 1.3, 5)
##    right_eyes = right_eye_detect.detectMultiScale(gray,1.3,5)
##    left_eyes = left_eye_detect.detectMultiScale(gray,1.3,5)
    
##    for (x,y,w,h) in right_eyes:
##        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
##
##        for (x,y,w,h) in left_eyes:
##            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
##            
##    for (x,y,w,h) in faces:
##        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    for (x,y,w,h) in hands:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("face",img)
    if cv2.waitKey(1) == ord('q'):
        break

#cam.release()
cv2.destroyAllWindows()
