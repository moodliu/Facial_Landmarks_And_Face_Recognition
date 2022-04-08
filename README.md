# Facial_Landmarks_And_Face_Recognition

# 臉部辨識與臉部特徵點

## 使用環境

```
windows 10 64bit 20H2
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
joblib==1.1.0
imutils==0.5.4
scikit-learn==0.19.1
```

## 使用步驟

### 樣本取樣

執行`face_record_with_face_align_new.py`，預設使用webcam鏡頭，
可透過編輯Line.29

```
cap = cv2.VideoCapture(0)
#0代表使用預設鏡頭，如有外接鏡頭則編號依序增加
#也可改成影片檔名，即可以影片進行樣本取樣
```

如需增加或減少樣本數可更改Line.33

```
num = 300
```

取樣完畢後會在`authorized_person`資料夾中生成對應的資料夾，並將取樣檔案放入該資料夾中

### 神經網路訓練

執行`face_train_model_modify.py`，訓練完成後的相關檔案會存放在`nn_model`資料夾

### 臉部辨識

執行`face_detect_recog_with_face_align.py`會依照訓練結果來進行臉部判斷，如辨識後權重有達一定程度會顯示該名稱，如未與訓練集中人員相似則會顯示"UNKNOW"

## 目前已知問題

`"File "專案資料夾(依檔案存放位置不同會有不同路徑)\face_record_with_face_align_new.py", line 75, in <module> Aligned_face = fa.align(frame, gray, d)"`

到`align`的source code `facealigner.py` 將Line.64&65 改為 `eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2.0, (leftEyeCenter[1] + rightEyeCenter[1]) // 2.0)`
