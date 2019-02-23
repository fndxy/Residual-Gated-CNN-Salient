import tensorflow as tf
import vgg16
import cv2
import numpy as np
from tensorflow.python import pywrap_tensorflow
img_size = 256
label_size = img_size
from tflearn.layers.conv import global_avg_pool
#from ConvLSTMCell import BasicConvLSTM
from signed_distance import signed_distance_function


class Model:
    def __init__(self):
        self.vgg = vgg16.Vgg16()

        self.input_holder = tf.placeholder(tf.float32, [5, img_size, img_size, 3])
        self.label_holder = tf.placeholder(tf.float32, [5,label_size,label_size, 2])



        

        self.contour_th = 1.5
        self.contour_weight = 0.0001

    def build_model(self):
	fea_dim = 128
        #build the VGG-16 model
        vgg = self.vgg
	vgg.build(self.input_holder)

	self.Fea_P5 = self.MSCA(vgg.conv5_3,name='Fea_P5')
	self.Fea_P4 = self.MSCA(vgg.conv4_3,name='Fea_P4')
	self.Fea_P3 = self.MSCA(vgg.conv3_3,name='Fea_P3')
	self.Fea_P2 = self.MSCA(vgg.conv2_2,name='Fea_P2')
	self.Fea_P1 = self.MSCA(vgg.conv1_2,name='Fea_P1')
	
	
	h_5 = self.Fea_P5
	h_4 = self.Fea_P4
	h_3 = self.Fea_P3
	h_2 = self.Fea_P2
	h_1 = self.Fea_P1


	P5_reshape = tf.image.resize_images(self.Fea_P5,[img_size,img_size])
	P4_reshape = tf.image.resize_images(self.Fea_P4,[img_size,img_size])
	P3_reshape = tf.image.resize_images(self.Fea_P3,[img_size,img_size])
	P2_reshape = tf.image.resize_images(self.Fea_P2,[img_size,img_size])

	concats = tf.concat([self.Fea_P1,P2_reshape,P3_reshape,P4_reshape,P5_reshape],axis = -1)
	

        concat_feature = tf.nn.relu(self.Conv_2d(concats, [3,3,fea_dim*5,fea_dim], 0.01, 'S1_1', padding='SAME'))
	concat_feature = self.Squeeze_excitation_layer(concat_feature, fea_dim, 4, 'fusion_SE')

	concat_feature_1 = tf.expand_dims(concat_feature[0],axis = 0)
        concat_feature_2 = tf.expand_dims(concat_feature[1],axis = 0)
        concat_feature_3 = tf.expand_dims(concat_feature[2],axis = 0)
        concat_feature_4 = tf.expand_dims(concat_feature[3],axis = 0)
        concat_feature_5 = tf.expand_dims(concat_feature[4],axis = 0)

        input_hidden_m1 = tf.zeros_like(concat_feature_1)
        input_feature_m1 = tf.zeros_like(concat_feature_1)

        gate_output_stage1 = self.info_passing(concat_feature_1,input_hidden_m1,input_feature_m1,'lstm_stage1',None)
	gate_output_stage2=  self.info_passing(concat_feature_2,gate_output_stage1,concat_feature_1,'lstm_stage1',True)
	gate_output_stage3=  self.info_passing(concat_feature_3,gate_output_stage2,concat_feature_2,'lstm_stage1',True)
	gate_output_stage4=  self.info_passing(concat_feature_4,gate_output_stage3,concat_feature_3,'lstm_stage1',True)
	gate_output_stage5=  self.info_passing(concat_feature_5,gate_output_stage4,concat_feature_4,'lstm_stage1',True)

	gate_lstm_1 = tf.concat([gate_output_stage1,gate_output_stage2,gate_output_stage3,gate_output_stage4,gate_output_stage5],axis = 0)
	lstm_out_1 = tf.nn.relu(concat_feature+gate_lstm_1)

	stage2_input_1 = tf.expand_dims(lstm_out_1[0],axis = 0)
	stage2_input_2 = tf.expand_dims(lstm_out_1[1],axis = 0)
	stage2_input_3 = tf.expand_dims(lstm_out_1[2],axis = 0)
	stage2_input_4 = tf.expand_dims(lstm_out_1[3],axis = 0)
	stage2_input_5 = tf.expand_dims(lstm_out_1[4],axis = 0)

	input_hidden_mi_1 = tf.zeros_like(stage2_input_1)
	input_feature_mi_1 = tf.zeros_like(stage2_input_1)
	gate_output_stage1 = self.info_passing(stage2_input_1,input_hidden_mi_1,input_feature_mi_1,'lstm_stage2',None)
	gate_output_stage2 = self.info_passing(stage2_input_2,gate_output_stage1,stage2_input_1,'lstm_stage2',True)
	gate_output_stage3 = self.info_passing(stage2_input_3,gate_output_stage2,stage2_input_2,'lstm_stage2',True)
	gate_output_stage4 = self.info_passing(stage2_input_4,gate_output_stage3,stage2_input_3,'lstm_stage2',True)
	gate_output_stage5 = self.info_passing(stage2_input_5,gate_output_stage4,stage2_input_4,'lstm_stage2',True)
	gate_lstm_2 = tf.concat([gate_output_stage1,gate_output_stage2,gate_output_stage3,gate_output_stage4,gate_output_stage5],axis = 0)

	lstm_out_2 = gate_lstm_2+gate_lstm_1	
	self.S1 = lstm_out_2
	
	self.Score = self.Conv_2d(self.S1, [1,1,fea_dim,2], 0.01, 'S1_lstm', padding='SAME')
	self.Prob = tf.nn.softmax(self.Score)

  	t1_score = self.Prob[0,:,:,0]
	t2_score = self.Prob[1,:,:,0]
	t3_score = self.Prob[2,:,:,0]
	t4_score = self.Prob[3,:,:,0]
	t5_score = self.Prob[4,:,:,0]


	
	t1_label = tf.py_func(signed_distance_function,[self.label_holder[0,:,:,0]],tf.float32)
	t2_label = tf.py_func(signed_distance_function,[self.label_holder[1,:,:,0]],tf.float32)
	t3_label = tf.py_func(signed_distance_function,[self.label_holder[2,:,:,0]],tf.float32)
	t4_label = tf.py_func(signed_distance_function,[self.label_holder[3,:,:,0]],tf.float32)
	t5_label = tf.py_func(signed_distance_function,[self.label_holder[4,:,:,0]],tf.float32)
	
	loss_edge1 = tf.expand_dims(t1_score*t1_label,axis = 0)
	loss_edge2 = tf.expand_dims(t2_score*t2_label,axis = 0)
	loss_edge3 = tf.expand_dims(t3_score*t3_label,axis = 0)
	loss_edge4 = tf.expand_dims(t4_score*t4_label,axis = 0)
	loss_edge5 = tf.expand_dims(t5_score*t5_label,axis = 0)

	self.loss_edge = tf.reduce_mean(tf.concat([loss_edge1,loss_edge2,loss_edge3,loss_edge4,loss_edge5],axis = 0))
        self.Loss_Mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Score,labels=self.label_holder))+self.loss_edge
                       

        self.accuracy = tf.reduce_mean(tf.abs(self.Prob-self.label_holder))
	
	
	


	
	

    def Conv_2d(self, input_, shape, stddev, name, padding='SAME'):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=shape,
                                initializer=tf.truncated_normal_initializer(stddev=stddev))

            conv = tf.nn.conv2d(input_, W, [1, 1, 1, 1], padding=padding)

            b = tf.Variable(tf.constant(0.0, shape=[shape[3]]), name='b')
            conv = tf.nn.bias_add(conv, b)

            return conv
    def Conv_atrous(self,input_,shape,stddev,name,rate,padding='SAME'):
	with tf.variable_scope(name) as scope:
		W = tf.get_variable('W',shape=shape,initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.atrous_conv2d(input_,W,rate = rate,padding=padding)
		b = tf.Variable(tf.constant(0.0, shape=[shape[3]]), name='b')
		conv = tf.nn.bias_add(conv, b)
		return conv

    def _max_pool(self, bottom, name=None):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME')
    def info_passing(self,bottom,m1_hidden,m1_feature,name=None,reuse=None):
        fea_dim = 128
        shape = bottom.get_shape().as_list()
        with tf.variable_scope(name,reuse=reuse):
                gate = tf.nn.sigmoid(self.Conv_2d(m1_feature, [3,3,shape[-1],fea_dim], 0.01, name+'gate', padding='SAME'))
                m1_ = self.Conv_2d(m1_hidden, [3,3,shape[-1],fea_dim], 0.01, name+'_m1_hidden', padding='SAME')
                current = self.Conv_2d(bottom, [3,3,shape[-1],fea_dim], 0.01, name+'_current', padding='SAME')
                return gate*m1_+current

    def MSCA(self,bottom,name=None):#Multiscale context aware feature extraction
	output_channel = 32
	shape = bottom.get_shape().as_list()
	
	dilation_1 = self.Conv_2d(bottom, [3,3,shape[-1],128], 0.01, name+'dilation_1', padding='SAME')
	dilation_3 = self.Conv_atrous(bottom,[3,3,shape[-1],output_channel],0.1,name+'dilation_3',3)
	dilation_5 = self.Conv_atrous(bottom,[3,3,shape[-1],output_channel],0.1,name+'dilation_5',5)
	dilation_7 = self.Conv_atrous(bottom,[3,3,shape[-1],output_channel],0.1,name+'dilation_7',7)
	dilation_9 = self.Conv_atrous(bottom,[3,3,shape[-1],output_channel],0.1,name+'dilation_9',9)
	concats = tf.concat([dilation_1,dilation_3,dilation_5,dilation_7,dilation_9],axis = -1)
	conv = self.Conv_2d(concats, [3,3,256,128], 0.01, name+'concats', padding='SAME')
	return tf.nn.relu(conv)

    def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
	    with tf.name_scope(layer_name) :
		squeeze = global_avg_pool(input_x)

		excitation = tf.layers.dense(squeeze, units=out_dim / ratio, name=layer_name+'_fully_connected1')#,kernel_initializer = tf.zeros_initializer())
		excitation = tf.nn.relu(excitation)
		excitation = tf.layers.dense(excitation, units=out_dim, name=layer_name+'_fully_connected2')#,kernel_initializer =tf.zeros_initializer())
		excitation = tf.nn.sigmoid(excitation)

		excitation = tf.reshape(excitation, [-1,1,1,out_dim])
		
		
		scale = input_x+input_x * excitation

		return scale

if __name__ == "__main__":

    img = cv2.imread("dataset/MSRA-B/image/0_1_1339.jpg")
    model = Model()
    model.build_model()
    varss = tf.trainable_variables()
