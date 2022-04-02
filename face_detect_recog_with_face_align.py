"""
This script is used to demonstrate face recognition using webcam
"""
from tensorflow.python.keras.models import model_from_json
import dlib
import cv2
import numpy as np
from sklearn.externals import joblib
from imutils.face_utils import FaceAligner
import os
# ================================PARAMETER============================================

# Set tolerance for face detection smaller means more tolerance for example -0.5 compared with 0
tolerance=-0.5
target_dir='authorized_person/'

# The directory of the trained neural net model
nn_model_dir='nn_model/'
hdf5_filename = 'face_recog_special_weights.hdf5'
json_filename = 'face_recog_special_arch.json'
labeldict_filename = 'label_dict_special.pkl'

# DLIB's model path for face pose predictor and deep neural network model
predictor_path='shape_predictor_68_face_landmarks.dat'
face_rec_model_path='dlib_face_recognition_resnet_model_v1.dat'

# .pkl file containing dictionary information about person's label corresponding with neural network output data
label_dict=joblib.load(nn_model_dir+labeldict_filename)
#print(label_dict)
# ====================================================================================

json_model_file=open(nn_model_dir+json_filename, 'r')
json_model = json_model_file.read()
json_model_file.close()

cnn_model = model_from_json(json_model)
cnn_model.load_weights(nn_model_dir+hdf5_filename)

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
fa = FaceAligner(sp)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faceAlign = frame.copy()
    
    faces,scores,idx = detector.run(frame, 0,tolerance)

    for i, d in enumerate(faces):
        #將人臉範圍標記
        cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), 2)
        Aligned_face = fa.align(frame, gray, d) 
        Aligned_faces,alig_scores,alig_idx = detector.run(Aligned_face, 0, 0)  
        for alig_i, alig_d in enumerate(Aligned_faces):
            #辨識當下的臉部特徵點
            Aligned_face_shape = sp(Aligned_face, alig_d)
            #計算臉部特徵點之歐式距離的值
            face_descriptor = np.array([facerec.compute_face_descriptor(Aligned_face, Aligned_face_shape)])
            #將計算出來的值 讓訓練完成的模組進行辨識
            prediction = cnn_model.predict(face_descriptor)
            #print(prediction)

            highest_proba=0
            counter=0
            # print prediction
            for prob in prediction[0]:
                if prob > highest_proba and prob >=0.1:
                    highest_proba=prob
                    label=counter
                    label_prob=prob
                    identity = label_dict[label]
                    #print(identity)

                if counter==(len(label_dict)-1) and highest_proba==0: #unknow
                    label= label_dict[counter]
                    label_prob=prob
                    identity=label
                    #print(identity)
                counter+=1
                
            if identity!='UNKNOWN':
                cv2.putText(frame,identity+'='+str(round((label_prob*100),2))+'%',(d.left(),  d.top()-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            else:
                cv2.putText(frame,'???',(d.left(),  d.top()-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    #cv2.namedWindow('face detect', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('face detect', frame)
    #cv2.imshow('face align', Aligned_face)
    #os.system("pause")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

# When everything done, release the capture
