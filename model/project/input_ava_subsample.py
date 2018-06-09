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
# flags you can input, you should run the code with the default data as long as you are on the server
flags.DEFINE_string("data_path", 'AVA-II', "Where your input data stored.")
flags.DEFINE_string("image_path", '/data/ece194n/AVA/keyframes', "Where your image stored")
flags.DEFINE_string("image_save_path", 'images', "Where your image saved")
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
	df=pd.read_csv(data_path+'/ava_train_v2.1new.csv',sep=',',header=None)
	train_data=df.values
	df=pd.read_csv(data_path+'/ava_val_v2.1new.csv',sep=",",header=None)
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
	#training data
	data=training_data
	image_path=training_image_path
	save_path=training_save_path
	# I set a set which used to chose labels
	number_set = set([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
	matrix_size = 0
	for i in range (data.shape[0]):
		if (int(data[i,6]) in number_set):
			matrix_size+=1
    # variables to limit the data size, basically I limited every label under 4000 datas since the limited space of the server
    # size of matrix
	matrix_size=matrix_size-88770-154932-30472-3148-298-5000-5000-5000-5000+1
	#print(matrix_size)
    # used to count number of data with label 1
	num_1 = 0
	# used to remove 1 from set
	num_1_remove = False
	num_8 = 0
	num_8_remove = False
	num_12 = 0
	num_12_remove = False
	num_11 = 0
	num_11_remove = False
	num_14_remove = False
	num_14 = 0
	# two np array used to save labels and images
	image_matrix = np.zeros((matrix_size,224,224,3))
	train_labels = np.zeros((matrix_size))
	image_index=0
	for i in range (data.shape[0]):
		if(i%1000==0):# print the number of image we are cropping
			print('Processing cropping for the ',i,'th training image.')
		if(data[i,6] not in number_set):# used to filter all the other labels out
			continue
		#used to remove labels
		if(num_11 >= 4000 and num_11_remove == False):
			number_set.remove(11)
			num_11_remove = True
		if(num_14 >= 4000 and num_14_remove == False):
			number_set.remove(14)
			num_14_remove = True
		if(num_12 >= 4000 and num_12_remove == False):
			number_set.remove(12)
			num_12_remove = True
		if(num_1 >= 4000 and num_1_remove == False):
			number_set.remove(1)
			num_1_remove = True
		if(num_8 >= 4000 and num_8_remove == False):
			number_set.remove(8)
			num_8_remove = True
		# used to count number of datas
		if(int(data[i,6]) == 11):
			num_11+=1
		if(int(data[i,6]) == 14):
			num_14+=1
		if(int(data[i,6]) == 12):
			num_12+=1
		if(int(data[i,6]) == 1):
			num_1+=1
		if(int(data[i,6]) == 8):
			num_8+=1
		# doing cropping
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
		im2 = im[int(float(data[i,3])*im.shape[0]):int(float(data[i,5])*im.shape[0]),int(float(data[i,2])*im.shape[1]):int(float(data[i,4])*im.shape[1]),:]
		im3 = cv2.resize(im2,(224,224),interpolation=cv2.INTER_CUBIC)
		if(image_index < matrix_size):
			image_matrix[image_index,:,:,:] = im3
			train_labels[image_index] = int(data[i,6])
			image_index+=1
		if(if_save):
			savepath = save_path+'/'+str(data[i,0])
			if not (os.path.exists(savepath)):
				os.makedirs(savepath)
			image_name = savepath+'/'+str(data[i,1])+"_"+str(data[i,2])+"_"+str(data[i,3])+'_'+str(data[i,4])+'_'+str(data[i,5])+".jpg"
			cv2.imwrite(image_name,im3)
	training_image_matrix = image_matrix

	#val_data all the same as train data
	data=val_data
	image_path = val_image_path
	save_path= val_save_path
	number_set = set([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
	matrix_size = 0
	for i in range (data.shape[0]):
		if (int(data[i,6]) in number_set):
			matrix_size+=1
	matrix_size=matrix_size-19147-2172-8000-8000-4500-429-976-1378-497-410+1
	num_1=0
	num_3=0
	num_4=0
	num_8=0
	num_10=0
	num_11=0
	num_14=0
	num_1_remove=False
	num_3_remove=False
	num_4_remove=False
	num_8_remove=False
	num_10_remove=False
	num_11_remove=False
	num_14_remove=False
	image_matrix = np.zeros((matrix_size,224,224,3))
	test_labels = np.zeros((matrix_size))
	image_index=0
	for i in range (data.shape[0]):
                if(i%1000==0):
                        print('Processing cropping for the ',i,'th testing image.')
                if(int(data[i,6]) not in number_set):
                        continue
                if(num_11 >= 500 and num_11_remove==False):
                        number_set.remove(11)
                        num_11_remove=True
                if(num_14 >= 500 and num_14_remove==False):
                        number_set.remove(14)
                        num_14_remove=True
                if(num_1 >= 500 and num_1_remove==False):
                        number_set.remove(1)
                        num_1_remove=True
                if(num_3>=500 and num_3_remove==False):
                        number_set.remove(3)
                        num_3_remove=True
                if(num_4 >= 500 and num_4_remove==False):
                        number_set.remove(4)
                        num_4_remove=True
                if(num_8 >= 500 and num_8_remove==False):
                        number_set.remove(8)
                        num_8_remove=True
                if(num_10 >= 500 and num_10_remove==False):
                        number_set.remove(10)
                        num_10_remove=True
                if(int(data[i,6]) == 11):
                        num_11+=1
                if(int(data[i,6]) == 14):
                        num_14+=1
                if(int(data[i,6]) == 1):
                        num_1+=1
                if(int(data[i,6]) == 3):
                        num_3+=1
                if(int(data[i,6]) == 4):
                        num_4+=1
                if(int(data[i,6]) == 8):
                        num_8+=1
                if(int(data[i,6]) == 10):
                        num_10+=1
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
                if(image_index < matrix_size):
                        image_matrix[image_index,:,:,:] = im3
                        test_labels[image_index] = int(data[i,6])
                        image_index+=1
                if(if_save):
                        savepath = save_path+'/'+str(data[i,0])
                        if not (os.path.exists(savepath)):
                                os.makedirs(savepath)
                        image_name = savepath+'/'+str(data[i,1])+"_"+str(data[i,2])+"_"+str(data[i,3])+'_'+str(data[i,4])+'_'+str(data[i,5])+".jpg"
                        cv2.imwrite(image_name,im3)
	print('Finished cropping of all ',i,' testing images!')
	testing_image_matrix = image_matrix
	training_image_matrix=0
	train_labels=0
	return (training_image_matrix,train_labels,testing_image_matrix,test_labels)
def main():
	(train_data,val_data) = input_data(data_path,image_path,save_path)
	(training_image_matrix,train_labels,val_image_matrix,val_labels)=read_image(train_data,val_data,data_path,image_path,save_path,False)
	np.savez_compressed('val',labels=val_labels,data=val_image_matrix)
	np.save('train_labels',train_labels)
	np.save('train_image_matrix',training_image_matrix)
	np.save('val_image_matrix',val_image_matrix)
if __name__ == "__main__":
	main()
