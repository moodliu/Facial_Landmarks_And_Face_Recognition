"""
This script is used to record person's facial feature data using dlib face shape model
and deep neural network model to extract face feature in 128-D vector
"""

import dlib
import cv2
import os
import numpy as np
from sklearn.externals import joblib
import time

from imutils import face_utils

# Set the output directory of the user's data
target_dir = 'authorized_person/'

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# DLIB's model path for face pose predictor and deep neural network model
predictor_path='shape_predictor_68_face_landmarks.dat'
face_rec_model_path='dlib_face_recognition_resnet_model_v1.dat'

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
filename = "10427115"
cap = cv2.VideoCapture(0)

temp_data=[]
continued = True
#num = int(input("Number of samples: " ))
num = 300
while (True):
    #name = str(input("What's your name? "))
    name = str(input("What's your name? "))
    directory=target_dir+name+'/'

    if os.path.exists(directory):
        print ('Name already exist! Try again!')
        continued = False
    else:
        os.makedirs(directory)
        continued = True

    if continued:
        break

count = 1
done = False
total_s = time.time()
while (continued):
    ret, frame = cap.read()
    start=time.time()
    dets,scores,idx = detector.run(frame, 0,0)

    for i, d in enumerate(dets):
        if len(idx)==1 and count < num :
            shape = sp(frame, d)

            draw_shape = face_utils.shape_to_np(shape) #畫landmarks點
            for (x, y) in draw_shape:
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            #cv2.imwrite("face_result/" + str(count) + ".jpg", frame)
            #count = count + 1

            #print(int(frame.shape[1]), int(frame.shape[0]))
            face_descriptor = np.array([facerec.compute_face_descriptor(frame, shape)])

            if len(temp_data)==0:
                temp_data=face_descriptor
            else:
                temp_data=np.append(temp_data,face_descriptor,axis=0)

            cv2.rectangle(frame,(d.left(),d.top()),(d.right(),d.bottom()),(255,0,0),2)

    scaled_frame=cv2.resize(frame,(int(frame.shape[1]/3),int(frame.shape[0]/3))) # 將原本frame縮小成1/3大小
    dets,scores,idx = detector.run(scaled_frame, 0,0)
    for i, d2 in enumerate(dets):
        if len(idx)==1:
            shape2 = sp(scaled_frame, d2)
            face_descriptor = np.array([facerec.compute_face_descriptor(scaled_frame, shape2)])
            temp_data=np.append(temp_data,face_descriptor,axis=0)

    if len(temp_data) >= num :
        print (temp_data)
        print (len(temp_data))
        total_e = time.time()
        print("total_time:", total_e - total_s )
        # Save the user's training data output to .pkl
        joblib.dump(temp_data,directory+'/face_descriptor.pkl')
        done = True
        break

    cv2.imshow('face detect',frame)

    if cv2.waitKey(1) & done :
        break
    delta=time.time()-start
    fps=float(1)/float(delta)
    print(int(fps))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()