# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 22:05:27 2018

@author: Moodliu
"""

"""
This script is used to demonstrate face recognition using webcam
"""
from keras.models import Sequential
from keras.models import model_from_json
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import SGD
import dlib
import cv2
import os
import csv
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.externals import joblib
import time
from imutils.face_utils import FaceAligner
from imutils import face_utils
import imutils
import time

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
check = [] # list to check 是否為本人
check_before_execute = False #第一次抓到人臉時確認是否為自己

roll_call_file = "test.csv" # 點名用的csv檔案
first_frame = True #第一張frame

def roll_call( roll_call_file, identity ) :  # 點名
    with open( roll_call_file, 'r+' , newline='') as csvfile:
      # 讀取 CSV 檔案內容
      dic = csv.DictReader(csvfile)
      with open('temp.csv', 'w+' , newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dic.fieldnames)
        writer.writeheader()
        #name = str(input('Enter student id : ') )
        name = identity
        for row in dic:
          if row['學號'] == name :
             row['未到'] = 'F'
             row['實到'] = 'T'
             writer.writerow(row)
          
          else :
            if row['未到'] == 'T':
               row['未到'] = 'T'
               row['實到'] = 'F'
            writer.writerow(row)
          print( row )
          
      csvfile.close()
    os.remove( roll_call_file )
    os.rename("temp.csv", roll_call_file )

def check_ident( check, identity ) :
    counter = 0
    for i in range(len(check)) :
        if identity == check[i] :
            counter += 1

    if counter >= 25 :
        #mark in csv file
        roll_call( roll_call_file, identity )
        input(" Press enter to continue ! ")
    check.clear() # init list

while (True):
    ret, frame = cap.read()
    faceAlign = frame.copy()
    start=time.time()

    dets,scores,idx = detector.run(frame, 0,tolerance)

    for i, d in enumerate(dets):
        cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), 2) # 將人臉先框起來
        faceAlign = fa.align(frame, frame, d)  # new photo , 由原本的圖片變成只有臉的圖片， photo size = 256 * 256
        dets, scores, idx = detector.run(faceAlign, 0, 0)  # new photo 還需要再 detect一次
        for i1, d1 in enumerate(dets):
            shape = sp(faceAlign, d1)  # mark 68_face_landmarks
            face_descriptor = np.array([facerec.compute_face_descriptor(faceAlign, shape)])
            prediction = cnn_model.predict_proba(face_descriptor)

            highest_proba=0
            counter=0
            # print prediction
            for prob in prediction[0]:
                if prob > highest_proba and prob >=0.1:
                    highest_proba=prob
                    label=counter
                    label_prob=prob
                    identity = label_dict[label]

                if counter ==(len(label_dict)-1) and highest_proba==0: # unknow
                    label= label_dict[counter]
                    label_prob=prob
                    identity=label

                counter+=1

            if identity!='UNKNOWN':
                if check_before_execute == False :
                    #cv2.putText(frame, "Are you " + identity + " ? ",(d.left(), d.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    if first_frame == False :
                        first_frame = True
                        ans = str(input(" T or F ? ") )
                        if ans == "T" or ans == 't' :
                            check.append(identity)
                            check_before_execute = True
                    else :
                        first_frame = False
                        cv2.putText(frame, "Are you " + identity + " ? ",(d.left(), d.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                else :
                    check.append(identity)
                    cv2.putText(frame, identity ,(d.left(), d.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.imshow('face detect', frame)
                    if len( check ) >= 30 :
                        check_ident(check, identity)
                        check_before_execute = False

            else:
                cv2.putText(frame,'???',(d.left(),  d.top()-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    cv2.namedWindow('face detect', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('face detect', frame)
    #cv2.imshow('face align', faceAlign)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

    #delta=time.time()-start
    #fps=float(1)/float(delta)
    #5print(int(fps))

# When everything done, release the capture