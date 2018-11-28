import cv2
import time
import matplotlib

matplotlib.use('tkagg')
import plotly

# plotly.tools.set_credentials_file(username='Nome_Usuario', api_key='Chave_Gerada_No_Site')
import plotly.plotly as py
import plotly.graph_objs as go


def detect_faces(cascade_classifier, img, scaleFactor=1.1):
    # convert img from bgr to gray
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(gray_img, scaleFactor=scaleFactor, minNeighbors=5)
    print("Face detectadas: ", len(faces))

    for (x, y, w, h) in faces:
        cv2.rectangle(gray_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return gray_img

# loading classifier HAAR
haar_face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_alt.xml')

# loading classifier LBP
lbp_face_cascade = cv2.CascadeClassifier('resources/lbpcascade_frontalface.xml')

# load image path
path_img = ('img/img1.jpg')
# path_img = ('img/img2.jpg')
# path_img = ('img/img3.jpg')

# read the image
img = cv2.imread(path_img)

# if you want to resize the image
# gray_img = cv2.resize(gray_img, (320, 380))

# ------------HAAR-----------
time_before_haar = time.time()

image_cascade = detect_faces(haar_face_cascade, img, scaleFactor=1.1)
cv2.imshow("Image HAAR", image_cascade)

time_after_haar = time.time()

time_haar = time_after_haar - time_before_haar
print("Processing Time HAAR: ", str(round(time_haar, 3)))

# ------------LBP-----------

time_before_lbp = time.time()

image_cascade = detect_faces(lbp_face_cascade, img, scaleFactor=1.1)
cv2.imshow("Image LBP", image_cascade)

time_after_lbp = time.time()

time_lbp = time_after_lbp - time_before_lbp
print("Processing Time LBP: " + str(round(time_lbp, 3)))

# Generating graphics
# data = [go.Bar(x=['Imagem com HAAR', 'Imagem com LBP'], y=[time_haar, time_lbp])]
# py.plot(data, filename='Comparação de Velocidade de Processamento')

# closes the windows when 0 is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
