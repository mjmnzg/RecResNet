import tensorflow as tf
from collections import OrderedDict
from adda.models.model import register_model_fn

@register_model_fn('resrecnet')
def resrecnet(inputs, scope = 'resrecnet', is_training=True, reuse=False, n_classes=4, prob=0.5):
    """
        Version of Residual Recurrent Network (RecResNet) in Tensorflow v1.9.
        
        Jiménez-Guarneros M., Gómez-Gil P. "Cross-subject classi-
        fication of cognitive loads using a recurrent-residual 
        deep network". IEEE Symposium Series on Computational Inte-
        lligence (IEEE SSCI 2017).
        
        Original implementation was performed in Lagsane library, but it is not commonly 
        used now.
    
    Parameters:
        inputs - placeholder input.
        scope - identifier to register network in execution.
        type_rnn - option "LSTM" or "GRU".
        is_training - flag to ability dropout regularizer.
        reuse - flag to ability reuse network.
        n_classes - output neurons.
    """
    layers = OrderedDict()
    
    with tf.variable_scope(scope, reuse=reuse):    
        # [1] Convolutional Layer 3D
        conv1 = tf.layers.conv3d(inputs, filters=16, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='valid', activation=tf.nn.tanh)
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
        # [2] Convolutional Layer 3D
        conv2 = tf.layers.conv3d(conv1, filters=16, kernel_size=(1, 3, 3), strides=(1, 2, 2), padding='valid', activation=tf.nn.tanh)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
        
        # [3] Residual block
        step1 = tf.layers.conv3d(conv2, filters=16, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='same', activation=None)
        step1 = tf.layers.batch_normalization(step1, training=is_training)
        step2 = tf.nn.relu(step1)
        step3 = tf.layers.conv3d(step2, filters=16, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='same', activation=None)
        step4 = tf.layers.batch_normalization(step3, training=is_training)
        residual = tf.nn.relu(conv2 + step4)
                
        # [4] Reshape layer
        nsamples = int(residual.get_shape()[1])
        nfeatures = int(residual.get_shape()[2]*residual.get_shape()[3]*residual.get_shape()[4])
        residual = tf.reshape(residual, [-1, nsamples, nfeatures])

        # default GRU
        cell = tf.contrib.rnn.GRUCell(128)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=prob)
        cell = tf.contrib.rnn.MultiRNNCell([cell])
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, residual, dtype=tf.float32)

        # [6] Flatten
        rnn_out = tf.contrib.layers.flatten(rnn_outputs)
        # [7] Fully connected layer
        fc1 = tf.contrib.layers.fully_connected(rnn_out, 100, activation_fn=tf.nn.relu)
        # [10] Dropout regularizer
        fc1 = tf.layers.dropout(fc1, rate=prob, training=is_training)
        # [8] Fully connected layer
        fc2 = tf.contrib.layers.fully_connected(fc1, 100, activation_fn=tf.nn.relu)
        # [10] Dropout regularizer
        layers["features"] = fc2 = tf.layers.dropout(fc2, rate=prob, training=is_training)
        # [11] Output Layer
        layers["output"] = tf.contrib.layers.fully_connected(fc2, n_classes, activation_fn=None)
        
    return layers
