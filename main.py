import cv2
import time

def detect_faces(cascade_classifier, img, scaleFactor=1.1):
    ### convert img from bgr to gray
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = cascade_classifier.detectMultiScale(gray_img, scaleFactor=scaleFactor, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(gray_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return gray_img
### loading cascade classifier HAAR
haar_face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_alt.xml')

### loading cascade classifier LBP
lbp_face_cascade = cv2.CascadeClassifier('resources/lbpcascade_frontalface.xml')

### load image path
path_img = ('img/img1.jpg')
# path_img = ('img/img2.jpg')
# path_img = ('img/img3.jpg')

### read the image
img = cv2.imread(path_img)

### convert img from rgb to gray
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

### resizig image
# gray_img = cv2.resize(gray_img, (320, 380))

###call function to detect faces - HAAR
faces_haar = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

###call function to detect faces - LBP
faces_lbp = lbp_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces_haar:
    cv2.rectangle(gray_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

###------------HAAR-----------
#time_before_haar = time.time()

#image_cascade = detect_faces(haar_face_cascade, img, scaleFactor=1.1)
#cv2.imshow("Image HAAR",image_cascade)

#time_after_haar = time.time()

#time_haar = time_after_haar - time_before_haar
#print("Processing Time HAAR: ", str(round(time_haar, 3)))

###------------LBP-----------
#time_before_lbp = time.time()

#image_cascade = detect_faces(lbp_face_cascade, img, scaleFactor=1.1)
#cv2.imshow("Image LBP",image_cascade)

#time_after_lbp = time.time()

#time_lbp = time_after_lbp - time_before_lbp
#print("Processing Time LBP: "+ str(round(time_lbp, 3)))

###---------------------------
#print("Face detectadas:  ", len(faces_haar))

# show img in window
cv2.imshow('Show Image', gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()