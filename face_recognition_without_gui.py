from tensorflow.python.keras.models import model_from_json
import dlib
import cv2
import numpy as np
from sklearn.externals import joblib
from imutils.face_utils import FaceAligner
import os
import csv


ROOT_DIR = os.getcwd()
datename = input("Date?")
predictor_path = './face_detect_landmarks_model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = './face_detect_landmarks_model/dlib_face_recognition_resnet_model_v1.dat'

hdf5_filename = 'face_recog_special_weights.hdf5'
json_filename = 'face_recog_special_arch.json'
labeldict_filename = 'label_dict_special.pkl'

MODEL_PATH = os.path.join(ROOT_DIR, "model")

nn_model_dir = os.path.join(ROOT_DIR, "nn_model")
nn_model_dir = nn_model_dir + "\\"

# .pkl file containing dictionary information about person's label corresponding with neural network output data
label_dict=joblib.load(nn_model_dir+labeldict_filename)
#print(label_dict)
json_model_file=open(nn_model_dir+json_filename, 'r')
json_model = json_model_file.read()
json_model_file.close()

cnn_model = model_from_json(json_model)
cnn_model.load_weights(nn_model_dir+hdf5_filename)

label_dict=joblib.load(nn_model_dir+labeldict_filename)

# 人臉辨識
detector = dlib.get_frontal_face_detector()
# 人臉特徵點辨識
predictor = dlib.shape_predictor(predictor_path)
# 人臉校正
fa = FaceAligner(predictor)
# 將人臉的資訊提取成一個128维的向量，如果臉部更相似，距離會更加接近、符合
# 使用歐幾里得距離來計算，公式:distence = sqrt((x1-x2)^2+(y1-y2)^2)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
cnn_model = model_from_json(json_model)
cnn_model.load_weights(nn_model_dir+hdf5_filename)


def face_predict(faces, gray, frame) :
    for d, face in enumerate(faces) :
        x = face.left()
        y = face.top()
        w = face.right()-face.left()
        h = face.bottom() - face.top()
        
        identity = ""
        Aligned_face = fa.align(frame, gray, face)
        Aligned_face = cv2.resize(Aligned_face, None,fx=0.5, fy=0.5)
        Aligned_faces, Alig_scores, Alig_idx = detector.run(
                Aligned_face, 0, 0)
        for alig_i, alig_d in enumerate(Aligned_faces):
                Aligned_face_shape = predictor(Aligned_face, alig_d)        
                face_descriptor = np.array(
                            [facerec.compute_face_descriptor(Aligned_face, Aligned_face_shape)])
                prediction = cnn_model.predict(face_descriptor)
                max_prob = np.max(prediction) 
                index = np.argmax(prediction)
                identity = label_dict[index]
                if identity == label_dict[len(label_dict) - 1] :
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                else :
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,str(identity) + " (" + str(max_prob) + ")",(x+5,y+h+15),cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,255,255), 2)
        return identity


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    # cap.set(6,cv2.VideoWriter.fourcc('M','J','P','G'))
    frame_count = 0
    current_identity = ""
    identity_list = []
    while(True) :
        
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces, scores, idx = detector.run(gray, 0, 0)
        if len(scores) > 0 :
            identity = face_predict(faces,gray,frame)
            if str(identity) != "" and str(identity) != "UNKNOWN" :
                if str(identity) in identity_list :
                    print("你已經點過囉!")

                elif str(identity) == current_identity :
                    frame_count = frame_count + 1
                    if frame_count >= 20 :
                        print("點名成功")
                        frame_count = 0 
                        identity_list.append(str(identity))
                
                else :
                    frame_count = 1
                    current_identity = str(identity)
        
        cv2.imshow("FaceShow", frame)
        if cv2.waitKey(1)  & 0xFF == ord('q'):
            break
    
    fp = open("./output/" + datename + ".csv","a")

    for name in label_dict :
        if name != "UNKNOWN" and str(label_dict[name]) in identity_list :
            fp.write(str(label_dict[name]) + " 1\n" )
        elif name!= "UNKNOWN" and not (str(label_dict[name]) in identity_list) :
            fp.write(str(label_dict[name]) + " 0\n")
    fp.close()
        
    

    