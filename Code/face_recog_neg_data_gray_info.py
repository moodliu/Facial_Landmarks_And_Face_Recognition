from sklearn.externals import joblib
temp_data = []
temp_data=joblib.load('unknown_person/preprocessed_data/face_recog_neg_data_gray.pkl')

print(len(temp_data))
with open("face_recog_neg_data_gray.txt", "a") as face_recog_neg_data:
    for i in range(len(temp_data)) :
        print(temp_data[i], file=face_recog_neg_data)
    face_recog_neg_data.close()
