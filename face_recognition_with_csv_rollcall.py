'''
Author       : Liu Xin-Yi
Date         : 2022-04-11 21:59:17
LastEditors  : Liu Xin-Yi
LastEditTime : 2022-04-29 09:13:20
FilePath     : face_recognition_with_csv_rollcall
Description  : 

Copyright (c) 2022 by Moodliu, All Rights Reserved.
'''
from tensorflow.python.keras.models import model_from_json
import dlib
import cv2
import numpy as np
from sklearn.externals import joblib
from imutils.face_utils import FaceAligner
import os
import csv
import glob

##########variable##########
detect_count = 0
current_id = ''
previous_id = ''
rollcalled_person = []
##########variable##########

##########Function##########


def load_csv():
    csv_list = glob.glob("./*.csv")
    print("資料夾中的CSV檔 : ")
    print(csv_list)
    target_csv = input('請輸入您要開啟的CSV檔案名稱(無須加附檔名) : ')
    with open(target_csv+'.csv', 'r+', newline='') as target_csv_file:
        dic = csv.DictReader(target_csv_file)
        dic_field, dic_list = dic.fieldnames, list(dic)
        target_csv_file.close()
    print('目標CSV檔載入成功 !')
    return dic_list, dic_field


def update_csv(dic_list, dic_field, student_id_list):
    with open('temp.csv', 'w+', newline='') as temp_csv_file:
        writer = csv.DictWriter(temp_csv_file, fieldnames=dic_field)
        writer.writeheader()
        for row in dic_list:
            for student_id in student_id_list:
                if row['學號'] == student_id:
                    row['未到'] = 'F'
                    row['實到'] = 'T'
            print(row)
            writer.writerow(row)
    temp_csv_file.close()


def check_same_person(current_id):
    # cv2.putText() color => BGR
    # detect_count_sum = 20
    global previous_id
    global detect_count
    global rollcalled_person
    if len(previous_id) == 0:
        # First detect or detect途中有不同人的情況
        previous_id = current_id
    if current_id == previous_id:
        detect_count = detect_count+1
        if detect_count == 20:  # 可調
            #'ID : '+current_id+'done !'+'('+str(round((label_prob*100),2))+'%)'
            # 將人臉範圍標記
            cv2.rectangle(frame, (d.left(), d.top()),
                          (d.right(), d.bottom()), (0, 255, 0), 2)
            # GREEN
            cv2.putText(frame, 'ID : '+current_id+' done !', (d.left(),
                        d.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            rollcalled_person.append(current_id)
            print('Student ID : ' + current_id + ' 點名成功')
        elif rollcalled_person.count(current_id) > 0:  # 已經點名成功過
            # 將人臉範圍標記
            cv2.rectangle(frame, (d.left(), d.top()),
                          (d.right(), d.bottom()), (0, 255, 0), 2)
            # GREEN
            cv2.putText(frame, 'ID : '+current_id+' already done !', (d.left(),
                        d.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            print('Student ID : ' + current_id + ' 已經點名了')
        else:
            # RED
            #'Detecting ID : '+current_id+'('+str(round((label_prob*100),2))+'%)'
            cv2.rectangle(frame, (d.left(), d.top()),
                          (d.right(), d.bottom()), (0, 0, 255), 2)
            cv2.putText(frame, 'Detecting ID : '+current_id, (d.left(),
                        d.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        # 辨識途中出現其他人 或unknown
        detect_count = 0
        previous_id = ''
##########Function##########


# Set tolerance for face detection smaller means more tolerance for example -0.5 compared with 0
tolerance = -0.5
target_dir = 'authorized_person/'

# The directory of the trained neural net model
nn_model_dir = 'nn_model/'
hdf5_filename = 'face_recog_special_weights.hdf5'
json_filename = 'face_recog_special_arch.json'
labeldict_filename = 'label_dict_special.pkl'
# .pkl file containing dictionary information about person's label corresponding with neural network output data
label_dict = joblib.load(nn_model_dir+labeldict_filename)
##########Load neural network information##########
json_model_file = open(nn_model_dir+json_filename, 'r')
json_model = json_model_file.read()
json_model_file.close()
nn_model = model_from_json(json_model)
nn_model.load_weights(nn_model_dir+hdf5_filename)
##########Load neural network information##########

##########DLIB's model path for face pose predictor and deep neural network model##########
predictor_path = './face_detect_landmarks_model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = './face_detect_landmarks_model/dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
fa = FaceAligner(sp)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
##########DLIB's model path for face pose predictor and deep neural network model##########

CSV_list, CSV_field = load_csv()
cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces, scores, idx = detector.run(gray, 0, tolerance)

    if len(scores) == 0:  # not detect face
        cv2.rectangle(frame, (int(frame.shape[1]/2)-120, int(frame.shape[0]/2)-120), (int(
            frame.shape[1]/2)+120, int(frame.shape[0]/2)+120), (255, 255, 0), 2)
        cv2.putText(frame, 'Detecting...', (int(frame.shape[1]/2)-120, int(
            frame.shape[0]/2)-130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    for i, d in enumerate(faces):
        # 將人臉範圍標記
        #cv2.rectangle(frame, (d.left(), d.top()),(d.right(), d.bottom()), (255, 0, 0), 2)
        Aligned_face = fa.align(frame, gray, d)
        Aligned_face = cv2.resize(Aligned_face, None, fx=0.5, fy=0.5)
        Aligned_face_gray = cv2.cvtColor(Aligned_face, cv2.COLOR_BGR2GRAY)
        Aligned_faces, alig_scores, alig_idx = detector.run(
            Aligned_face_gray, 0, 0)
        for alig_i, alig_d in enumerate(Aligned_faces):
            # 辨識當下的臉部特徵點
            Aligned_face_shape = sp(Aligned_face, alig_d)
            # 計算臉部特徵點之歐式距離的值
            face_descriptor = np.array(
                [facerec.compute_face_descriptor(Aligned_face, Aligned_face_shape)])
            # 將計算出來的值 讓訓練完成的模組進行辨識
            prediction = nn_model.predict(face_descriptor)
            # 會顯示face_descriptor對model每個lable的相似程度，value介於[0,1]
            # print(prediction)

            highest_proba = 0
            counter = 0
            # print prediction
            for prob in prediction[0]:
                # 跟model進行比對
                if prob > highest_proba and prob >= 0.1:
                    highest_proba = prob
                    label = counter
                    # label_prob cv2.putText 用
                    #label_prob = prob
                    identity = label_dict[label]

                if counter == (len(label_dict)-1) and highest_proba == 0:  # unknow
                    label = label_dict[counter]
                    # label_prob cv2.putText 用
                    #label_prob = prob
                    identity = label
                counter += 1

            if identity != 'UNKNOWN':
                current_id = identity
                check_same_person(current_id)
            else:
                cv2.rectangle(frame, (d.left(), d.top()),
                              (d.right(), d.bottom()), (0, 0, 0), 2)
                cv2.putText(frame, '???', (d.left(),  d.top()-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # BLACK
                detect_count = 0

    cv2.imshow('face detect', frame)
    #cv2.imshow('face align', Aligned_face)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Writing rollcall data in csv file .')
        update_csv(CSV_list, CSV_field, rollcalled_person)
        print('Finish writing job .')
        cap.release()
        cv2.destroyAllWindows()
        break

# When everything done, release the capture
