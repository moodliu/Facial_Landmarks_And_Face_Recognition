import dlib
import cv2
import os
import numpy as np
from sklearn.externals import joblib
import time
from imutils.face_utils import FaceAligner
from imutils import face_utils
import imutils


predictor_path = 'shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
fa = FaceAligner(sp)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

cap = cv2.VideoCapture(0)

while True :
    ret, frame = cap.read()
    copy_frame = frame.copy()

    dets, scores, idx = detector.run(copy_frame, 0, 0)
    for i,d in enumerate(dets) :
        faceAlign = fa.align(copy_frame, copy_frame, d) # new photo
        dets, scores, idx = detector.run(faceAlign, 0, 0) # new photo 還需要再 detect一次
        for i1,d1 in enumerate(dets) :
            shape = sp(faceAlign, d1)
            draw_shape = face_utils.shape_to_np(shape) #畫landmarks點
            for (x, y) in draw_shape:
                cv2.circle(faceAlign, (x, y), 1, (0, 0, 255), -1)
                
    
    dets1, scores1, idx1 = detector.run(frame, 0, 0)
    for i1, d1 in enumerate(dets1):
        shape = sp(frame, d1)
        draw_shape = face_utils.shape_to_np(shape)  #畫landmarks點
        for (x, y) in draw_shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow('frame', frame)
    
    
    cv2.imshow('faceAlign', faceAlign)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()