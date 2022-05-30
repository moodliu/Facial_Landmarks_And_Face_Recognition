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
python = 3.6
Nvidia版本 = 511.23
Cuda版本 = 10.2
cmake = 3.13.2
```

## 所需套件

### 可透過以下指令安裝

```text
pip install requirements.txt #裡面不包含DLIB，如有需要請另下指令安裝，GPU版本在安裝時會有些許複雜
```

### 主要所需套件

```text
tensorflow-gpu==2.3.0
Keras==2.3.1
opencv-python==4.5.5.62
dlib==19.22.0 #Default use CPU
cmake==3.22.1
numpy==1.18.5
imutils==0.5.4
scikit-learn==0.19.1
```

### 以下列出Dlib安裝成功時Visual Studio的配置，不保證都是必須安裝的元件

```text
Visual C++ 核心桌面功能
VC++ 2017 version 15.9 v14.16 latest v141 tools
C++分析工具
Windows 10 SDK (10.0.17763.0)
適用於CMake的Visual C++工具
x86與x64版C++譯器AT
MSBuild
桌上型電腦版的VC++ 2015.3 v14.00(140)工具組
英文語言套件
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

[方法一](https://github.com/PyImageSearch/imutils/issues/254) `https://github.com/PyImageSearch/imutils/issues/254`

[方法二](https://blog.csdn.net/m0_46825740/article/details/120429730) `https://blog.csdn.net/m0_46825740/article/details/120429730`

## 貼心體醒

### Dlib 安裝方法

#### setup.py 安裝

`Dlib` 較新版本的可從官網下載->解壓縮->開CMD->cd到目的地資料夾-> python setup.py install 成功安裝後就可使用GPU功能 預設為使用CPU。

#### pip 安裝(未試驗過)

`pip install dlib -v`

### Visual Studio Installer 配置(在網路上看到的)

`個別元件 -> Compilers,build tools,and runtimes -> VC++ ... toolset`

### 安裝完成後測試

```text
import dlib
print(dlib.DLIB_USE_CUDA) # True 表示可以使用 GPU
```
