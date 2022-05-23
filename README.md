# Facial_Landmarks_And_Face_Recognition

# 臉部辨識與臉部特徵點

## 使用環境

```
windows 10 64bit 20H2
Visual studio 2017
python=3.6
Nvidia版本=511.23
Cuda版本=10.2
```

## 所需套件

### 可透過以下指令安裝

```
pip install requirements.txt
```

### 主要所需套件

```
tensorflow-gpu==2.3.0
Keras==2.3.1
opencv-python==4.5.5.62
dlib==19.22.0
cmake==3.22.1
numpy==1.18.5
imutils==0.5.4
scikit-learn==0.19.1
```

## 使用步驟

### 樣本取樣

有 webcam版本`face_record_with_webcam.py`

與影片版本`face_record_with_video.py`，使用前須將影片檔案放入`videos`資料夾內

如需改變樣本數可更改Line.47(face_record_with_video.py)或
Line.53(face_record_with_webcam.py)

```
sample_count= 300
```

取樣完畢後會在`authorized_person`資料夾中生成對應的資料夾，並將取樣檔案放入該資料夾中

使用`face_record_with_video`會依照檔案名稱生成對應的取樣檔案，如:test.mp4->temp.pkl

使用`face_record_with_webcam` 會依照執行時輸入的名稱去對取樣檔案進行命名

### 神經網路訓練

執行`face_train_model.py`，訓練完成後的相關檔案會存放在`nn_model`資料夾

### 臉部辨識(待更新)

執行`face_recognition_with_csv_rollcall.py`會依照訓練結果來進行臉部判斷，如辨識後權重有達一定程度會顯示該名稱，如未與訓練集中人員相似則會顯示"UNKNOW"

## 目前已知問題

如發生以下報錯(待更新)

```
"File "專案資料夾(依檔案存放位置不同會有不同路徑)\face_record_with_face_align_new.py", line 75, in <module> Aligned_face = fa.align(frame, gray, d)"
File "C:\Users\(Username)\anaconda3\lib\site-packages\imutils\face_utils\facealigner.py", line 68, in align
M = cv2.getRotationMatrix2D(eyesCenter, angle, , scale)
TypeError: Can't parse 'center'. Sequence item with index 0 has a wrong type
```

到`align`的source code `facealigner.py` 將Line.64&65 改為 `eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2.0, (leftEyeCenter[1] + rightEyeCenter[1]) // 2.0)`

## 貼心體醒

`Dlib` 較新版本的可從官網下載->解壓縮->開CMD->cd到目的地資料夾->pip install cmake -> python setup.py install 就可使用GPU功能 預設為使用CPU。

```----------------#Dlib GPU ----------------
import dlib
print(dlib.DLIB_USE_CUDA) # True 表示可以使用 GPU
#(舊版本19.8.1要安裝GPU版本需複雜操作，不建議)
#(以取樣300次比較，約快4倍時間)
```
