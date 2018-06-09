from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2 #used to read image
flags=tf.flags
logging=tf.logging
flags.DEFINE_string("data_path", 'AVA-II', "Where your input data stored. Usually AVA-II")
flags.DEFINE_string("image_path", '/data/ece194n/AVA/keyframes', "Where your image stored. Usually /data/ece194n/AVA/keyframes/")
flags.DEFINE_string("image_save_path", 'images', "Where your image saved. Usually images")
FLAGS = flags.FLAGS
data_path=FLAGS.data_path
data_path=data_path.strip()
data_path=data_path.rstrip("/")
image_path=FLAGS.image_path
image_path=image_path.strip()
image_path=image_path.rstrip("/")
save_path=FLAGS.image_save_path
save_path=save_path.strip()
save_path=save_path.rstrip("/")
def input_data(data_path,image_path,save_path):
	df=pd.read_csv(data_path+'/ava_train.csv',sep=',',header=None)
	train_data=df.values
	df=pd.read_csv(data_path+'/ava_val_v2.csv',sep=",",header=None)
	val_data=df.values
	return (train_data,val_data)
def read_image(training_data,val_data,data_path,image_path,save_path,if_save=False):
	training_image_path = image_path+'/train_keyframes/'
	val_image_path = image_path+'/val_keyframes/'
	training_save_path = save_path+"/train_images"
	val_save_path = save_path+"/val_images"
	# mkdir
	if not (os.path.exists(save_path)):
		os.makedirs(save_path)
	if not (os.path.exists(training_save_path)):
		os.makedirs(training_save_path)
	if not (os.path.exists(val_save_path)):
		os.makedirs(val_save_path)
	#valid_number = set([1,2,3,4,5,6,7,8,9,10,11,12,13]) #actions we chose
	#training data
	data=training_data
	image_path=training_image_path
	save_path=training_save_path
	image_matrix = np.zeros((data.shape[0],224,224,3))
	for i in range (data.shape[0]):
		if(i%1000==0):
			print('Processing cropping for the ',i,'th training image.')
		image_file = str(data[i,0])
		image_time = str(data[i,1])
		imageyu = float(data[i,2])
		imagexu = float(data[i,3])
		imageyd = float(data[i,4])
		imagexd = float(data[i,5])
		imagefeature = int(data[i,6])
		if(len(image_time)==3):
			image_time = '0'+image_time
		imagepath = image_path+'/'+image_file+'/'+image_time.rstrip()+'.jpg'
		im = cv2.imread(imagepath).astype(np.float64)
		#print(imagepath)
		im2 = im[int(float(data[i,3])*im.shape[0]):int(float(data[i,5])*im.shape[0]),int(float(data[i,2])*im.shape[1]):int(float(data[i,4])*im.shape[1]),:]
		#print(im2.shape)
		im3 = cv2.resize(im2,(224,224),interpolation=cv2.INTER_CUBIC)
		image_matrix[i,:,:,:] = im3
		if(if_save):
			savepath = save_path+'/'+str(data[i,0])
			if not (os.path.exists(savepath)):
				os.makedirs(savepath)
			image_name = savepath+'/'+str(data[i,1])+"_"+str(data[i,2])+"_"+str(data[i,3])+'_'+str(data[i,4])+'_'+str(data[i,5])+".jpg"
			cv2.imwrite(image_name,im3)
	training_image_matrix = image_matrix
	#val_data
	data=val_data
	image_path = val_image_path
	save_path= val_save_path
	image_matrix = np.zeros((data.shape[0],224,224,3))
	for i in range (data.shape[0]):
                if(i%1000==0):
                        print('Processing cropping for the ',i,'th testing image.')
                image_file = str(data[i,0])
                image_time = str(data[i,1])
                imageyu = float(data[i,2])
                imagexu = float(data[i,3])
                imageyd = float(data[i,4])
                imagexd = float(data[i,5])
                imagefeature = int(data[i,6])
                if(len(image_time)==3):
                        image_time = '0'+image_time
                imagepath = image_path+'/'+image_file+'/'+image_time.rstrip()+'.jpg'
                im = cv2.imread(imagepath).astype(np.float64)
                #print(imagepath)
                im2 = im[int(float(data[i,3])*im.shape[0]):int(float(data[i,5])*im.shape[0]),int(float(data[i,2])*im.shape[1]):int(float(data[i,4])*im.shape[1]),:]
                #print(im2.shape)
                im3 = cv2.resize(im2,(224,224),interpolation=cv2.INTER_CUBIC)
                image_matrix[i,:,:,:] = im3
                if(if_save):
                        savepath = save_path+'/'+str(data[i,0])
                        if not (os.path.exists(savepath)):
                                os.makedirs(savepath)
                        image_name = savepath+'/'+str(data[i,1])+"_"+str(data[i,2])+"_"+str(data[i,3])+'_'+str(data[i,4])+'_'+str(data[i,5])+".jpg"
                        cv2.imwrite(image_name,im3)
	print('Finished cropping of all ',i,' testing images!')
	testing_image_matrix = image_matrix
	return (training_image_matrix,testing_image_matrix)
def main():
	#print('haha')
	(train_data,val_data) = input_data(data_path,image_path,save_path)
	(training_image_matrix,val_image_matrix)=read_image(train_data,val_data,data_path,image_path,save_path)
	train_labels = train_data[:,6]
	val_labels = val_data[:,6]
	np.save('val_labels',val_labels)
	np.save('train_labels',train_labels)
	np.save('train_image_matrix',training_image_matrix)
	np.save('val_image_matrix',val_image_matrix)
if __name__ == "__main__":
	main()
