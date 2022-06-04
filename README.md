<!--
 * @Author       : Liu Xin-Yi
 * @Date         : 2022-05-17 10:23:52
 * @LastEditors  : Liu Xin-Yi
 * @LastEditTime : 2022-06-01 13:50:18
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
適用於Cmake和Linux的Visual C++ 工具
x86與x64版C++譯器AT
MSBuild
桌上型電腦版的VC++ 2015.3 v14.00(140)工具組
英文語言套件
```

### 需在環境變數->path額外新增的路徑

`C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64`

`C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64\cl.exe`

#### Dlib安裝步驟

下載Dlib zip(19.22.0) -> 解壓縮 -> 開啟CMD(系統管理員) -> 選擇要安裝的虛擬環境 -> cd 到dlib資料夾 ->
`mkdir build` -> `cd build` ->
`cmake -G "Visual Studio 15 2017 Win64" -T host=x64 .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1` ->
`cmake --build .` #注意最後有一個 `.`  ->
`cd..` -> `python setup.py install`  #在過程中會有很多Warning，那好像是因為文字編碼的問題導致，但不影響安裝

##### ***注意事項***

* Visual Studio版本需介於VS2015到VS2019間，有試過在VS2022安裝，會跳出錯誤，原因是因為CUDA資料夾中`host_config.h`的設定而導致。
* 如果上述步驟沒有問題，但在執行`python setup.py install`有無法安裝的狀況，遇到的問題大多是因為系統內`libgif`檔案照不到或損壞，可以改成下列指令`python setup.py install --no DLIB_GIF_SUPPORT`

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

### Dlib安裝完成後測試

```text
import dlib
print(dlib.DLIB_USE_CUDA) # True 表示可以使用 GPU
```

### 程式報錯問題

如在`face_descriptor = np.array([facerec.compute_face_descriptor(Aligned_face, Aligned_face_shape)])`有發生以下報錯：

```text
發生例外狀況:RuntimeError
Error while calling cublasCreate(&handles[new_device_id]) in file (Dlib安裝路徑)\dlib\cuda\cublas_dlibapi.cpp:78. code: 1, reason: CUDA Runtime API initialization
```

推測是因為tensorflow在啟動時會佔據大部分的GPU memory，導致在執行時可用的剩餘視訊記憶體不足，而無法完成程式碼所需的初始化作業，可在程式碼上半部分加入以下程式碼：

```text
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
```
