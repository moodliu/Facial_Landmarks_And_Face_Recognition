'''
Author       : Liu Xin-Yi
Date         : 2022-04-09 11:30:56
LastEditors  : Liu Xin-Yi
LastEditTime : 2022-05-26 11:22:37
FilePath     : face_record_with_webcam
Description  : 

Copyright (c) 2022 by Moodliu, All Rights Reserved.
'''
import os
import dlib
import cv2
import numpy as np
from sklearn.externals import joblib
from imutils.face_utils import FaceAligner
import time

def proccess_percent(cur, total):
    if cur+1 == total:
        percent = 100.0
        print('Sample Extraction Processing : %5s [%d/%d]' %
              (str(percent)+'%', cur+1, total), end='\n')
    else:
        percent = round(1.0*cur/total*100, 1)
        print('Sample Extraction Processing : %5s [%d/%d]' %
              (str(percent)+'%', cur+1, total), end='\r')

# Set the output directory of the user's data
target_dir = './authorized_person/'

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# DLIB's model path for face pose predictor and deep neural network model
predictor_path='./face_detect_landmarks_model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path='./face_detect_landmarks_model/dlib_face_recognition_resnet_model_v1.dat'

#人臉辨識
detector = dlib.get_frontal_face_detector() 
#人臉特徵點辨識
predictor = dlib.shape_predictor(predictor_path) 
#人臉校正
fa = FaceAligner(predictor)
#將人臉的資訊提取成一個128维的向量，如果臉部更相似，距離會更加接近、符合
#使用歐幾里得距離來計算，公式:distence = sqrt((x1-x2)^2+(y1-y2)^2)
facerec = dlib.face_recognition_model_v1(face_rec_model_path) 

cap = cv2.VideoCapture(0)
temp_data=[]
continued = True
sample_count = 300

#資料夾創建
while (True):
    name = str(input("What's your name? "))
    #name = filename
    directory=target_dir+name+'/'

    if os.path.exists(directory):
        print ('Name already exist! Try again!')
        continued = False
    else:
        os.makedirs(directory)
        continued = True

    if continued:
        break

done = False
start = time.clock()
while (continued):
    #ret boolean ,判斷是否有擷取到影像
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    #圖片灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #scores代表辨識分數，分數越高則人臉辨識的精確率越高，而idx代表臉部方向
    #第三個參數是指定分數的門檻值，所有分數超過這個門檻值的偵測結果都會被輸出
    faces,scores,idx = detector.run(gray, 0,0.8)

    for i,d in enumerate(faces) :
        '''
        Original_face_shape = predictor(frame, d)
        #畫landmarks點
        Original_face_draw_shape = face_utils.shape_to_np(Original_face_shape)  
        for (x, y) in Original_face_draw_shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        '''
        Aligned_face = fa.align(frame, gray, d) 
        Aligned_face = cv2.resize(Aligned_face, None, fx=0.5, fy=0.5)
        Aligned_faces,Alig_scores,Alig_idx = detector.run(Aligned_face, 0,0)
        for alig_i, alig_d in enumerate(Aligned_faces) :
            Aligned_face_shape = predictor(Aligned_face, alig_d)
            '''
            #畫landmarks點
            Aligned_face_draw_shape = face_utils.shape_to_np(Aligned_face_shape)  
            for (x, y) in Aligned_face_draw_shape:
                cv2.circle(Aligned_face, (x, y), 1, (0, 0, 255), -1)
            '''
            face_descriptor = np.array([facerec.compute_face_descriptor(Aligned_face, Aligned_face_shape)])    
            if len(temp_data)==0:
                temp_data=face_descriptor
            else:
                temp_data=np.append(temp_data,face_descriptor,axis=0)
        proccess_percent(len(temp_data), sample_count)  
        cv2.imshow('Original',frame)
        cv2.imshow('Aligned',Aligned_face)

    if len(temp_data) >= sample_count :
        # Save the user's training data output to .pkl
        joblib.dump(temp_data,directory+'/face_descriptor.pkl')
        done = True
        continued = False

    if cv2.waitKey(1) & done :
        break
elapsed = (time.clock() - start)
print("Time used:", elapsed)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()