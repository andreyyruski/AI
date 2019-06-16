import cv2
import time

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
profil_face_cascade = cv2.CascadeClassifier("haarcascade_profileface.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

#image = cv2.imread("faces.jpg")

#image = cv2.resize(image, (int(image.shape[1]/7), int(image.shape[0]/7)))

def detectElements(image, cascade, scale = 1.2, Neighbors = 5, color = (0, 255, 0)):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elements = cascade.detectMultiScale(gray_image, scaleFactor = scale, minNeighbors = Neighbors)

    for x, y, w, h in elements:
        image = cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)


while True:

    start_time = time.time()
    check, frame = video.read()

    detectElements(frame, face_cascade, 1.3)
    detectElements(frame, profil_face_cascade, 1.3, color = (0, 100, 0))
    detectElements(frame, eye_cascade, color = (255, 0, 0))
#    detectElements(frame, smile_cascade, color = (0, 0, 255))

# add fps on the screen
    fps = round(1 / (time.time() - start_time), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,"fps - " + str(fps),(0,50), font, 1,(0,255,0),2,cv2.LINE_AA)

    cv2.imshow("Faces & Eyes", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
