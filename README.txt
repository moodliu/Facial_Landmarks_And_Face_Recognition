tensorflow安裝：pip install tensorflow-gpu

Dlib 較新版本的可從官網下載->解壓縮->開CMD->cd到目的地資料夾->pip install cmake -> python setup.py install 就可使用GPU功能 預設為使用CPU。
----------------#Dlib GPU ----------------
		import dlib
		dlib.DLIB_USE_CUDA # True 表示可以使用 GPU   
(舊版本19.8.1要安裝GPU版本需複雜操作，不建議)
(以取樣300次比較，約快4倍時間)

執行過程中如出現
"File "C:\path\專題-original\face_record_with_face_align_new.py", line 75, in <module>
    Aligned_face = fa.align(frame, gray, d)"
到align的source code "facealigner.py" line.64&65 改為 eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2.0, (leftEyeCenter[1] + rightEyeCenter[1]) // 2.0)