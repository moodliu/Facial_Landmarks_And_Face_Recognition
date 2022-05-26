<!--
 * @Author       : Liu Xin-Yi
 * @Date         : 2022-05-17 10:23:52
 * @LastEditors  : Liu Xin-Yi
 * @LastEditTime : 2022-05-24 23:42:42
 * @FilePath     : README
 * @Description  : 
 * 
 * Copyright (c) 2022 by Moodliu, All Rights Reserved.
-->

# Facial_Landmarks_And_Face_Recognition

# 臉部辨識與臉部特徵點

## 使用環境

```text
windows 10 64bit 20H2
Visual studio 2017
python=3.6
Nvidia版本=511.23
Cuda版本=10.2
```

## 所需套件

### 可透過以下指令安裝

```text
pip install requirements.txt
```

### 主要所需套件

```text
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

與影片版本`face_record_with_video.py`，使用前須將影片檔案放入`videos`資料夾內。

如需改變樣本數可更改Line.47(face_record_with_video.py)或
Line.53(face_record_with_webcam.py)。

`sample_count = 300`

取樣完畢後會在`authorized_person`資料夾中生成對應的資料夾，並將取樣檔案放入該資料夾中，

使用`face_record_with_video`會依照檔案名稱生成對應的樣本檔案，如:test.mp4->temp.pkl，

使用`face_record_with_webcam` 會依照執行時輸入的名稱去對樣本檔案進行命名。

### 神經網路訓練

執行`face_train_model.py`，訓練完成後的相關檔案會存放在`nn_model`資料夾。

### 臉部辨識

這個程式會依照訓練模組結果來進行臉部判斷，如辨識後權重有達一定程度會顯示該名稱。


#### 使用步驟介紹

在執行`face_recognition_with_csv_rollcall.py`前須先將點名需要的檔案(csv檔)放到目前的資料夾中，

在執行時需要先依照引導在Terminal中輸入點名檔案的名稱，

而在辨識完成後按`Q`則會結束程式，並將辨識的結果寫入檔案中。

#### 辨識畫面介紹


如果沒有偵測到人臉時，會在畫面中央顯示色碼為`#00FFFF`的方框並在上方顯示`Detecting...`，

在辨識階段則會在人臉位置出現色碼為`#FF0000`的方框並在上方顯示`Detecting ID : (辨識到的人員名稱)`，

當該人員點名完成時，會在人臉位置出現色碼為`#00FF00`的方框並在上方顯示`ID : (辨識到的人員名稱) done!`並在Terminal中顯示`Student ID : (辨識到的人員名稱) 點名成功`，

而在該人員點名完成後再次被辨識到時，則會在人臉位置出現色碼為`#00FF00`的方框並在上方顯示`ID : (辨識到的人員名稱) already done!`並在Terminal中顯示`Student ID : (辨識到的人員名稱) 已經點名過了`，

如果辨識結果無法匹配到訓練結果中的任一人，則會在人臉位置出現色碼為`#000000`的方框並在上方顯示`???`。

## 目前已知問題

如發生以下報錯

```text
Traceback(most recent call last):
  "File "專案資料夾(依檔案存放位置不同會有不同路徑)\face_record_with_face_align_new.py", line 75, in <module>   Aligned_face = fa.align(frame, gray, d)"
  File "C:\Users\(Username)\anaconda3\lib\site-packages\imutils\face_utils\facealigner.py", line 68, in align
  M = cv2.getRotationMatrix2D(eyesCenter, angle, , scale)
TypeError: Can't parse 'center'. Sequence item with index 0 has a wrong type
```

可以參考以下兩種解決方法

`https://github.com/PyImageSearch/imutils/issues/254`

`https://blog.csdn.net/m0_46825740/article/details/120429730`

## 貼心體醒(待更新)

`Dlib` 較新版本的可從官網下載->解壓縮->開CMD->cd到目的地資料夾-> python setup.py install 成功安裝後就可使用GPU功能 預設為使用CPU。

安裝VS CMAKE...

```----------------#Dlib GPU ----------------
import dlib
print(dlib.DLIB_USE_CUDA) # True 表示可以使用 GPU
#(舊版本19.8.1要安裝GPU版本需複雜操作，不建議)
#(以取樣300次比較，約快4倍時間)
```
