import os
import numpy as np
from sklearn.externals import joblib
pos_image_dir='authorized_person/'
authorized_person_list=os.listdir(pos_image_dir)
for person in authorized_person_list:       
    temp_data=joblib.load(pos_image_dir+person+'/face_descriptor.pkl')
    for i in range(len(temp_data)) :
        with open(person + "_face_descriptor_info.txt", "a") as face_descriptor_info:
            print(temp_data[i], file=face_descriptor_info)
            face_descriptor_info.close()
    