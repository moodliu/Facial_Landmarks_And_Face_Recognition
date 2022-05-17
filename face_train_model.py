"""
This script is used to train the model to recognize person's ID based on it's facial feature
using Keras

The output of this script is stored in nn_model directory which will contain .hdf5, .json and .pkl files

The .hdf5 file is the keras weight file
The .json file is the keras architecture file
The .pkl file is the file containing dictionary information about person's label corresponding with neural network output data
"""

import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.externals import joblib
from tensorflow.python.keras.models import Sequential, model_from_json
#from keras import backend as K
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.utils import np_utils
#from keras.utils import plot_model
from tensorflow.keras.optimizers import SGD,RMSprop, Adam

# Enable continue_training to train existing model if set to 'True'
continue_training = False

# Ratio of Train and Test data
#70% Data for training 30% Data for test
test_data_ratio=0.3

# The output directory of the trained neural net model
nn_model_dir='nn_model/'
hdf5_filename = 'face_recog_special_weights.hdf5'
json_filename = 'face_recog_special_arch.json'
labeldict_filename = 'label_dict_special.pkl'

# The input directory of positive and negative person's facial data
pos_image_dir='authorized_person/'
neg_image_dir='unknown_person/'

# Keras neural network model learning parameter
batch_size = 50
epochs = 15
loss = 'categorical_crossentropy'
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
optimizer=adam

def make_model(nb_class):
	print('Building NN architecture model..')
	model = Sequential()
	model.add(Dense(4096,input_dim=128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_class))
	model.add(Activation('softmax'))
	#plot_model(model, to_file='Network.png', show_shapes=True, show_layer_names= True )
	model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])
	
	print('Finished building NN architecture..')
	return model

def train_model(model, train_data, train_label, test_data, test_label, nb_epoch):
	
	checkpointer = ModelCheckpoint(filepath=nn_model_dir+hdf5_filename,
								   verbose=1,
								   save_best_only=True)
	
	#command = tensorboard --logdir=logs
	#tensorboard = TensorBoard(log_dir='logs', write_graph=True, write_images=True)

	cnn_json_model = model.to_json()

	with open(nn_model_dir+json_filename, "w") as json_file:
		json_file.write(cnn_json_model)
	
	print("Saved NN architecture to disk..")
		
	print('Start optimizing NN model..')
	model.fit(train_data,
			  train_label,
			  batch_size=batch_size,
			  epochs =nb_epoch,
			  validation_data=(test_data, test_label),
			  callbacks=[checkpointer],
			  shuffle=True,
			  verbose=1)
	
	print('Optimization finished..')
	
	return model

x_train=[]
x_test=[]
y_train=[]
y_test=[]

authorized_person_list=os.listdir(pos_image_dir)
authorized_person_list.remove('.gitignore')
#加上一加上一個unknow person的lable
nb_class=len(authorized_person_list)+1

print ('Building neural network architecture...')
if not continue_training:
	cnn_model = make_model(nb_class)
else:
	json_model_file=open(nn_model_dir+json_filename, 'r')
	json_model = json_model_file.read()
	json_model_file.close()

	cnn_model = model_from_json(json_model)

	cnn_model.load_weights(nn_model_dir+hdf5_filename)

	cnn_model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])

print ('Processing AUTHORIZED person data...')

label_dict=dict()
class_counter=0
for person in authorized_person_list:
	print ('Processing %s data.....'%(person))
	label_dict[class_counter]=person
	temp_data=joblib.load(pos_image_dir+person+'/face_descriptor.pkl')
	temp_label=np.repeat(class_counter,len(temp_data))

	temp_x_train, temp_x_test, temp_y_train, temp_y_test = train_test_split(
		temp_data, temp_label, test_size=test_data_ratio, random_state=42)
	print ('Obtained %i train and %i test data'%(len(temp_x_train),len(temp_x_test)))
	if len(x_train)==0:
		x_train=temp_x_train
		x_test=temp_x_test
		y_train=np.append(y_train,temp_y_train)
		y_test=np.append(y_test,temp_y_test)
	else:
		x_train=np.append(x_train,temp_x_train,axis=0)
		x_test=np.append(x_test,temp_x_test,axis=0)
		y_train=np.append(y_train,temp_y_train)
		y_test=np.append(y_test,temp_y_test)
	class_counter+=1

print ('Finished...')

print ('Processing UNKNOWN person data')
label_dict[class_counter]='UNKNOWN'
joblib.dump(label_dict,nn_model_dir+labeldict_filename)

neg_data=joblib.load(neg_image_dir+'preprocessed_data/unknown_person_face_descriptor.pkl')
temp_data=neg_data

temp_label=np.repeat(class_counter,len(temp_data))

temp_x_train, temp_x_test, temp_y_train, temp_y_test = train_test_split(
	temp_data, temp_label, test_size=test_data_ratio, random_state=42)
	
x_train=np.append(x_train,temp_x_train,axis=0)
x_test=np.append(x_test,temp_x_test,axis=0)
y_train=np.append(y_train,temp_y_train)
y_test=np.append(y_test,temp_y_test)

print ('Finished extracting data.....')

y_train_cat = y_train.astype('int')
y_train_cat = np_utils.to_categorical(y_train_cat, nb_class)

y_test_cat = y_test.astype('int')
y_test_cat = np_utils.to_categorical(y_test_cat, nb_class)

trained_cnn_model = train_model(cnn_model, x_train, y_train_cat, x_test, y_test_cat, epochs)
