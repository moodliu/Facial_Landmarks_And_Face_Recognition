from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import numpy as np
import imutils
import dlib
import cv2
 
detector = dlib.get_frontal_face_detector()
predictor_path='shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
fa = FaceAligner(predictor, desiredFaceWidth=200)
 
face_filename = 1
def detect_face_landmarks(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
 
    for face in faces:
        (x, y, w, h) = rect_to_bb(face)
        faceOrig = imutils.resize(img[y : y+h, x : x+w], width=200)
        faceAligned = fa.align(img, gray, face)
        global face_filename
        cv2.imwrite('./faceOrig_{0}.png'.format(face_filename), faceOrig)
        cv2.imwrite('./faceAligned_{0}.png'.format(face_filename), faceAligned)
        face_filename += 1
 
 
filename = "P1200487.JPG"
detect_face_landmarks(filename)