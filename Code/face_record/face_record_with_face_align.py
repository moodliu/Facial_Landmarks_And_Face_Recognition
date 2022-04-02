import os
import dlib
import cv2
import numpy as np
from sklearn.externals import joblib
import time
from imutils.face_utils import FaceAligner
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
fa = FaceAligner(sp)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

#filename = "10427273"
cap = cv2.VideoCapture('10427109.mp4')

temp_data=[]
continued = True
#num = int(input("Number of samples: " ))
num = 300

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


count = 0
done = False
total_s = time.time()
while (continued):
    ret, frame = cap.read()

    start=time.time()
    dets,scores,idx = detector.run(frame, 0,0)

    for i,d in enumerate(dets) :
        faceAlign = fa.align(frame, frame, d)  # new photo (256*256)
        dets, scores, idx = detector.run(faceAlign, 0, 0) # new photo 還需要再 detect一次
        frame = faceAlign
        for i1,d1 in enumerate(dets) :
            if len(idx)==1 and count < num :
                shape = sp(faceAlign, d1)
                draw_shape = face_utils.shape_to_np(shape)  #畫landmarks點
                for (x, y) in draw_shape:
                    cv2.circle(faceAlign, (x, y), 1, (0, 0, 255), -1)
                face_descriptor = np.array(
                    [facerec.compute_face_descriptor(faceAlign, shape)])

                if len(temp_data)==0:
                    temp_data=face_descriptor
                else:
                    temp_data=np.append(temp_data,face_descriptor,axis=0)



    if len(temp_data) >= num :
        #print (temp_data)
        #print (len(temp_data))
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