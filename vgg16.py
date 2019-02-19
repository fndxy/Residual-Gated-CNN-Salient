import os
import sys

import numpy as np
import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
VGG_MEAN = [103.939, 116.779, 123.68]

# https://github.com/machrisaa/tensorflow-vgg

class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = sys.modules[self.__class__.__module__].__file__
            # print path
            path = os.path.abspath(os.path.join(path, os.pardir))
            # print path
            path = os.path.join(path, "vgg16.npy")
            print(path)
            vgg16_npy_path = path

        self.data_dict = np.load('vgg16_weights.npz')
        print("npy file loaded")

    def build(self, input, train=False):

        self.conv1_1 = self._conv_layer(input, "conv1_1")
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
	
        self.pool1 = self._max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
	
        self.pool2 = self._max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
	
        self.pool3 = self._max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
	
        self.pool4 = self._max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
	
        self.pool5 = self._max_pool(self.conv5_3, 'pool5')


    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def DPP_sym_lite(self,feature,name):
	with tf.variable_scope(name):
		feature_shape = feature.get_shape().as_list()
		I = feature
		It = tf.image.resize_images(tf.nn.avg_pool(I,ksize = [1,2,2,1],strides = [1,2,2,1],padding='SAME'),[feature_shape[1],feature_shape[2]],method = 1)
		
		x = tf.add(tf.square(tf.subtract(I,It)),1e-3)
		xn = tf.image.resize_images(tf.nn.avg_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding='SAME'),[feature_shape[1],feature_shape[2]],method=1)

		weights_lambda = tf.get_variable('_lambda',xn.get_shape().as_list(),initializer=tf.random_normal_initializer(stddev=0.1))
		_lambda = tf.exp(weights_lambda)
		weights_alpha = tf.get_variable('_alpha',xn.get_shape().as_list(),initializer=tf.random_normal_initializer(stddev=0.1))
		_alpha = tf.exp(weights_alpha)

		w = tf.pow(x/xn,_lambda)+_alpha
		kp = tf.nn.avg_pool(w,ksize = [1,2,2,1],strides = [1,2,2,1],padding='SAME')
			
		Iw = tf.nn.avg_pool(tf.multiply(I,w),ksize = [1,2,2,1],strides = [1,2,2,1],padding='SAME')
		output = tf.div(Iw,kp)
		return output

    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def _fc_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):

        #W_regul = lambda x: self.L2(x)

        #return tf.get_variable(name="filter",
        #                       initializer=self.data_dict[name][0],
        #                       trainable=True,
        #                       regularizer=W_regul)
        return tf.Variable(self.data_dict[name+'_W'], name="filter")

    def get_bias(self, name):
        return tf.Variable(self.data_dict[name+'_b'], name="biases")

    def get_fc_weight(self, name):
        return tf.Variable(self.data_dict[name+'_W'], name="weights")

    def L2(self, tensor, wd=0.001):
        return tf.mul(tf.nn.l2_loss(tensor), wd, name='L2-Loss')

