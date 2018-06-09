File description:

input_ava.py: Used to generate the input numpy array.
input_ava_subsample.py: Used to generate the input numpy array with subsampling
resnet50_AVA.py: The resnet50 run on AVA data set.
resnet50_mnist.py The resnet50 run on mnist data set.
resnet101_AVA.py: The resnet101 run on AVA data set.
resnet101_mnist.py: The resnet101 run on mnist data set.
label_predict.py: Used to predict labels

Code running instruction:
First run input_ava.py:
python3 input_ava.py --data_path=AVA-II --image_path=/data/ece194n/AVA/keyframes --image_save_path=images
Then run resnet50_AVA.py to train the model:
CUDA_VISIBLE_DEVICES=0 python3 resnet50_AVA.py
Then run labels_predict.py to see the output accuracy
