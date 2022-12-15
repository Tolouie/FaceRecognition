import cv2

cascPath = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread('picture.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cascPath.detectMultiScale(gray,1.1,4)

#Draw rectangle around faces
for(x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (225,0,0), 2)

cv2.imshow('img', img)
cv2.waitKey()

cv2.imwrite("face_detected.jpg", img)