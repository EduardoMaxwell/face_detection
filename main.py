import cv2
import time

#convert img from gray scale to RGB
def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#load image path
path_img = ('bird_img.jpeg')

bird_img = cv2.imread(path_img) #read the image
gray_bird_img = cv2.cvtColor(bird_img, cv2.COLOR_BGR2GRAY) #convert img from rgb to gray

cv2.imshow('Show Image', gray_bird_img) #show img in window
cv2.waitKey(0)
cv2.destroyAllWindows()

