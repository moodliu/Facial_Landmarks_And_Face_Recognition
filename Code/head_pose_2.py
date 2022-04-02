# 把一些警告的訊息暫時関掉
import warnings
warnings.filterwarnings('ignore')

# Utilities相關函式庫
import os
import numpy as np
import math

# 圖像處理/展現的相關函式庫
import cv2
import dlib
import matplotlib.pyplot as plt

SHAPE_MODEL_PATH = "shape_predictor_68_face_landmarks.dat"

# 將dlib偵測到的人臉68個特徵點取出
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


# 偵測單一人臉的臉部特徵(假設圖像中只有一個人)
def get_landmarks(im, face_detector, shape_predictor):
    rects = face_detector(im, 1)
    shape = shape_predictor(im, rects[0])
    coords = shape_to_np(shape, dtype="int")

    return coords

# 使用dlib自帶的frontal_face_detector作為我們的人臉偵測器
face_detector = dlib.get_frontal_face_detector()
face_landmark_predictor = dlib.shape_predictor(SHAPE_MODEL_PATH)

cap = cv2.VideoCapture(0)

while True :
    ret, frame = cap.read()
    copy = frame.copy()
    size = frame.shape

    dets = face_detector(frame, 1)

    if ( len(dets) > 0 ) :
        # 取得單1人臉的68個人臉關鍵點的座標
        landmarks = get_landmarks(frame, face_detector, face_landmark_predictor)
        # 鼻尖 Nose tip: 34
        nose_tip = landmarks[33:34]
        # 下巴 Chin: 9
        chin = landmarks[8:9]
        # 左眼左角 Left eye left corner: 37
        left_eye_corner = landmarks[36:37]
        # 右眼右角 Right eye right corner: 46
        right_eye_corner = landmarks[45:46]
        # 嘴巴左角 Left Mouth corner: 49
        left_mouth_corner = landmarks[48:49]
        # 嘴巴右角 Right Mouth corner: 55
        right_mouth_corner = landmarks[54:55]
        # 把相關的6個座標串接起來
        face_points = np.concatenate((nose_tip, chin, left_eye_corner, right_eye_corner, left_mouth_corner,right_mouth_corner))
        face_points = face_points.astype(np.double)

        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner                         
        ])

        # 焦距
        focal_length = size[1]
        #print("Cameria [focal_length]: ", focal_length)

        # 照像機內部成像的中心點(w, h)
        center = (size[1] / 2, size[0] / 2)

        # 照像機參數 (Camera internals )
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],dtype="double")

        #print("Camera Matrix :\n {0}".format(camera_matrix))

        # 扭曲係數
        dist_coeffs = np.zeros((4, 1))  # 假設沒有鏡頭的成像扭曲 (no lens distortion)

        # 使用OpenCV的solvePnP函數來計算人臉的旋轉與位移
        #(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix
        #                                                              , dist_coeffs, flags=cv2.CV_ITERATIVE)
        # 參數:
        #   model_points 3維模型的座標點
        #   image_points 2維圖像的座標點
        #   camera_matrix 照像機矩陣
        #   dist_coeffs 照像機扭曲係數
        #   flags: cv2.SOLVEPNP_ITERATIVE
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points,face_points,camera_matrix,dist_coeffs,flags=cv2.SOLVEPNP_ITERATIVE)

        #print("Rotation Vector:\n {0}".format(rotation_vector))  # 旋轉向量
        #print("Translation Vector:\n {0}".format(translation_vector))  # 位移向量

        # 計算歐拉角
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]

        yaw = eulerAngles[1]
        pitch = eulerAngles[0]
        roll = eulerAngles[2]

        if pitch > 0:
            pitch = 180 - pitch
        elif pitch < 0:
            pitch = -180 - pitch
        yaw = -yaw

        # print("[pitch]: ", pitch)  # 抬頭(+)/低頭(-)
        # print("[yaw]  : ", yaw)  # 右轉(+)/左轉(-)
        # print("[roll] : ", roll)  # 右傾(+)/左傾(-)

        # 投射一個3D的點 (100.0, 0, 0)到2D圖像的座標上
        (x_end_point2D, jacobian) = cv2.projectPoints(np.array([(100.0, 0.0, 0.0)]), rotation_vector
                                                    , translation_vector, camera_matrix, dist_coeffs)

        # 投射一個3D的點 (0, 100.0, 0)到2D圖像的座標上
        (y_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 100.0, 0.0)]), rotation_vector
                                                    , translation_vector, camera_matrix, dist_coeffs)

        # 投射一個3D的點 (0, 0, 100.0)到2D圖像的座標上
        (z_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 100.0)]), rotation_vector
                                            , translation_vector, camera_matrix, dist_coeffs)

        # 以 Nose tip為中心點畫出x, y, z的軸線
        p_nose = (int(face_points[0][0]), int(face_points[0][1]))
        p_x = (int(x_end_point2D[0][0][0]), int(x_end_point2D[0][0][1]))
        p_y = (int(y_end_point2D[0][0][0]), int(y_end_point2D[0][0][1]))
        p_z = (int(z_end_point2D[0][0][0]), int(z_end_point2D[0][0][1]))

        cv2.line(frame, p_nose, p_x, (0,0,255), 3)  # X軸 (紅色)
        cv2.line(frame, p_nose, p_y, (0,255,0), 3)  # Y軸 (綠色)
        cv2.line(frame, p_nose, p_z, (255,0,0), 3)  # Z軸 (藍色)

        cv2.putText(
            frame,
            "X: " + str(pitch), (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 0, 0),
            thickness=2)

        cv2.putText(
            frame,
            "Y: " + str(yaw), (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 0, 0),
            thickness=2)

        cv2.putText(
            frame,
            "Z: " + str(roll), (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 0, 0),
            thickness=2)

        # 把6個基準點標註出來
        for p in face_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (255,255,255), -1)

    cv2.imshow('face detect', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
