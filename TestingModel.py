import cv2
import numpy as np
import stage_passing 
import os
import sys
import tensorflow as tf
import time
import vgg16
import random
from skimage import io,transform


def load_img_list(dataset):

    if dataset == 'MSRA-B':
        path = 'dataset/MSRA-B/image'
    elif dataset == 'HKU-IS':
        path = 'dataset/HKU-IS/imgs'
    elif dataset == 'DUT-OMRON':
        path = 'dataset/DUT-OMRON/DUT-OMRON-image'
    elif dataset == 'PASCAL-S':
        path = 'dataset/PASCAL-S/pascal'
    elif dataset == 'SOD':
        path = 'dataset/BSDS300/imgs'
    elif dataset == 'ECSSD':
        path = 'dataset/ECSSD/images'

    imgs = os.listdir(path)

    return path, imgs


trainval_set = ['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'drift-chicane', 'drift-straight', 'goat', 'horsejump-high', 'kite-surf', 'libby', 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black', 'soapbox']



annotation_dir = './video_data/DAVIS/Annotations/480p/'
imgset_dir = './video_data/DAVIS/JPEGImages/480p/'


def generate_trainingSet():
	img_pairs = []
	gt_pairs = []
	for dataset in trainval_set:
		training_list = imgset_dir+dataset+'/'
		annotation_list = annotation_dir+dataset+'/'
		
		train_img_list = os.listdir(training_list)
		
		ann_img_list = os.listdir(annotation_list)
		train_img_list.sort()
		

		for i in range(4,len(train_img_list)):
			pair_img1 = training_list+train_img_list[i-4]
			pair_img2 = training_list+train_img_list[i-3]
			pair_img3 = training_list+train_img_list[i-2]
			pair_img4 = training_list+train_img_list[i-1]
			pair_img5 = training_list+train_img_list[i]
			img_pairs.append([pair_img1,pair_img2,pair_img3,pair_img4,pair_img5])
			
	return img_pairs








if __name__ == "__main__":

    model = stage_passing.Model()
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    img_size = stage_passing.img_size
    label_size = stage_passing.label_size

    ckpt = tf.train.get_checkpoint_state('./model_spatialtemporal/')
    saver = tf.train.Saver()
    var_list = tf.trainable_variables()


    
    saver.restore(sess, ckpt.model_checkpoint_path)


    img_pairs = generate_trainingSet()

    
    for f_img in img_pairs:
	origin = cv2.imread(f_img[0])
	input_img1 =  cv2.resize(cv2.imread(f_img[0]),(img_size,img_size))*1.0- vgg16.VGG_MEAN
	input_img2 =  cv2.resize(cv2.imread(f_img[1]),(img_size,img_size))*1.0- vgg16.VGG_MEAN
	input_img3 =  cv2.resize(cv2.imread(f_img[2]),(img_size,img_size))*1.0- vgg16.VGG_MEAN
	input_img4 =  cv2.resize(cv2.imread(f_img[3]),(img_size,img_size))*1.0- vgg16.VGG_MEAN
	input_img5 =  cv2.resize(cv2.imread(f_img[4]),(img_size,img_size))*1.0- vgg16.VGG_MEAN

	img = np.stack((input_img1,input_img2,input_img3,input_img4,input_img5),axis = 0)
	result = sess.run(model.Prob,feed_dict={model.input_holder:img})
	result = result[-1,:,:,0]
	origin = cv2.imread(f_img[4])
	origin_shape = origin.shape
	
	result = result*255
	result = cv2.resize(result.astype(np.uint8),(origin.shape[1],origin.shape[0]))
	cv2.imshow('s',result)
	cv2.waitKey(0)
	

    
