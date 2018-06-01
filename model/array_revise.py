#THIS FILE IS USED TO SHRINK THE DATASETS
import numpy as np
a = np.load('val_labels.npy')
print('a finish loading')
aa = np.load('val_image_matrix.npy')
print('aa finish loading')
b = np.load('train_labels.npy')
print('b finish loading')
bb = np.load('train_image_matrix.npy')
print('bb finish loading')
a = a[0:1000]
aa = aa[0:1000,:,:,:]
b = b[0:1000]
bb = bb[0:1000,:,:,:]
np.save('val_labels_small',a)
print('a finish saving')
np.save('val_image_matrix_small',aa)
print('aa finish saving')
np.save('train_labels_small',b)
print('b finish saving')
np.save('train_image_matrix',bb)
print('bb finish saving')
