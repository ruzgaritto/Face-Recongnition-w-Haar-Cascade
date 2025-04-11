import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier("haar_cascade.xml")

features = np.load("features.npy", allow_pickle=True)
labels = np.load("labels.npy")

people = ["Ben Afflek", "Elton John", "Jerry Seinfield", "Madonna", "Mindy Kaling"]

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

img = cv.imread("/Users/ahmet/Python Projects/denemesahasi/Resources/Faces/val/madonna/1.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect face
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y + h, x:x + w]
    label, confidence = face_recognizer.predict(faces_roi)
    
    print(f"Label: {label}, Confidence: {confidence}")
    
    cv.putText(img, f"{people[label]}", (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow("Detected Faces", img)

cv.waitKey(0)
cv.destroyAllWindows()
