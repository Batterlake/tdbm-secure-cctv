import cv2
import numpy as np


def imread(path: str) -> np.ndarray:
    return cv2.imread(path)


def initialize(cascasdepath="haarcascade_frontalface_default.xml"):
    return cv2.CascadeClassifier(cascasdepath)


def face_detect(image: np.ndarray, face_cascade):
    result = image.copy()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + h, y + h), (0, 255, 0), 2)

    return result
