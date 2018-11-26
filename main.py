import cv2
import time

### loading cascade classifier
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
lbp_face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

### load image path
path_img = ('img/img1.jpg')
# path_img = ('img/img2.jpg')
# path_img = ('img/img3.jpg')

### read the image
bird_img = cv2.imread(path_img)

### convert img from rgb to gray
gray_img = cv2.cvtColor(bird_img, cv2.COLOR_BGR2GRAY)

### resizig image
# gray_bird_img = cv2.resize(gray_bird_img, (320, 380))

faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    cv2.rectangle(gray_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# I've tryied to do that, but it returns the same gray img
# img_colorful = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

print("Face detectadas:  ", len(faces))

# show img in window
cv2.imshow('Show Image', gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
