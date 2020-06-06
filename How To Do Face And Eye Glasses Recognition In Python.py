import cv2
import os


# Cascades
face_cascade = cv2.CascadeClassifier(os.path.expanduser('~/Desktop/Chatbot5/haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.expanduser('~/Desktop/Chatbot5/haarcascade_eye_tree_eyeglasses.xml'))


# Video Capture
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    cv2.imshow('face/eye_recognition', img)
    k = cv2.waitKey(30) and 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
