import dlib
import cv2
import numpy as np
from imutils import face_utils
from imutils.face_utils import FaceAligner

# DLIB's model path for face pose predictor and deep neural network model
predictor_path='shape_predictor_68_face_landmarks.dat'
face_rec_model_path='dlib_face_recognition_resnet_model_v1.dat'

#人臉辨識
detector = dlib.get_frontal_face_detector() 
#人臉特徵點辨識
predictor = dlib.shape_predictor(predictor_path) 
#人臉校正
fa = FaceAligner(predictor)
#將人臉的資訊提取成一個128维的向量，如果臉部更相似，距離會更加接近、符合
#使用歐幾里得距離來計算，公式:distence = sqrt((x1-x2)^2+(y1-y2)^2)
facerec = dlib.face_recognition_model_v1(face_rec_model_path) 

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while(True) :
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces,scores,idx = detector.run(frame, 0,0)
    for i,d in enumerate(faces) :
        Original_face_shape = predictor(frame, d)
        #畫landmarks點
        Original_face_draw_shape = face_utils.shape_to_np(Original_face_shape)  
        for (x, y) in Original_face_draw_shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        Aligned_face = fa.align(frame, gray, d) 
        Aligned_faces,Alig_scores,Alig_idx = detector.run(Aligned_face, 0,0)
        for alig_i, alig_d in enumerate(Aligned_faces) :
            Aligned_face_shape = predictor(Aligned_face, alig_d)
            '''
            #畫landmarks點
            Aligned_face_draw_shape = face_utils.shape_to_np(Aligned_face_shape)  
            for (x, y) in Aligned_face_draw_shape:
                cv2.circle(Aligned_face, (x, y), 1, (0, 0, 255), -1)
            '''
        cv2.imshow('Original',frame)
        cv2.imshow('Aligned',Aligned_face)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
# When everything done, release the capture
