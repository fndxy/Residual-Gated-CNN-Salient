import skfmm
import numpy as np
import tensorflow as tf


def signed_distance_function(matrix):
	k= -1*(skfmm.distance(matrix)-skfmm.distance(1.-matrix))
	return k.astype(np.float32)





#input_holder = tf.placeholder(tf.float32,(5,5))
#contour = tf.py_func(signed_distance_function,[input_holder],tf.float32)
#sess = tf.Session()
#print sess.run(contour,feed_dict={input_holder:tests})
