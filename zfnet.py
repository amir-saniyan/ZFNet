# -*- coding: utf-8 -*-

# TODO: Local contrast normalization.

import tensorflow as tf


class ZFNet:

    def __init__(self, input_width=224, input_height=224, input_channels=3, num_classes=1000, learning_rate=0.01,
                 momentum=0.9, keep_prob=0.5):

        # From article: Stochastic gradient descent with a mini-batch size of 128 was used to update the parameters,
        # starting with a learning rate of 10**−2, in conjunction with a momentum term of 0.9.
        # From article: Dropout is used in the fully connected layers (6 and 7) with a rate of 0.5.

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.keep_prob = keep_prob

        self.random_mean = 0
        self.random_stddev = 0.01

        # ----------------------------------------------------------------------------------------------------

        # Input: 224x224x3.
        with tf.name_scope('input'):
            self.X = tf.placeholder(dtype=tf.float32,
                                    shape=[None, self.input_height, self.input_width, self.input_channels], name='X')

        # Labels: 1000.
        with tf.name_scope('labels'):
            self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name='Y')

        # Dropout keep prob.
        with tf.name_scope('dropout'):
            self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='dropout_keep_prob')

        # Layer 1.
        # [Input] ==> 224x224x3
        # --> 224x224x3 ==> [Convolution: size=(7x7x3)x96, strides=2, padding=valid] ==> 110x110x96
        # --> 110x110x96 ==> [ReLU] ==> 110x110x96
        # --> 110x110x96 ==> [Max-Pool: size=3x3, strides=2, padding=valid] ==> 55x55x96
        # --> [Output] ==> 55x55x96
        # Note: There were some calculation errors in ZFNet architecture:
        # floor((224-7)/2) + 1 = 109
        # floor((110-3)/2) + 1 = 54
        with tf.name_scope('layer1'):
            layer1_activations = self.__conv(input=self.X, filter_width=7, filter_height=7, filters_count=96,
                                             stride_x=2, stride_y=2, padding='VALID')
            layer1_pool = self.__max_pool(input=layer1_activations, filter_width=3, filter_height=3, stride_x=2,
                                          stride_y=2, padding='VALID')

        # Layer 2.
        # [Input] ==> 55x55x96
        # --> 55x55x96 ==> [Convolution: size=(5x5x96)x256, strides=2, padding=valid] ==> 26x26x256
        # --> 26x26x256 ==> [ReLU] ==> 26x26x256
        # --> 26x26x256 ==> [Max-Pool: size=3x3, strides=2, padding=valid] ==> 13x13x256
        # --> [Output] ==> 13x13x256
        # Note: There were some calculation errors in ZFNet architecture:
        # floor((26-3)/2) + 1 = 12
        with tf.name_scope('layer2'):
            layer2_activations = self.__conv(input=layer1_pool, filter_width=5, filter_height=5, filters_count=256,
                                             stride_x=2, stride_y=2, padding='VALID')
            layer2_pool = self.__max_pool(input=layer2_activations, filter_width=3, filter_height=3, stride_x=2,
                                          stride_y=2, padding='VALID')

        # Layer 3.
        # [Input] ==> 13x13x256
        # --> 13x13x256 ==> [Convolution: size=(3x3x256)x384, strides=1, padding=same] ==> 13x13x384
        # --> 13x13x384 ==> [ReLU] ==> 13x13x384
        # --> [Output] ==> 13x13x384
        with tf.name_scope('layer3'):
            layer3_activations = self.__conv(input=layer2_pool, filter_width=3, filter_height=3, filters_count=384,
                                             stride_x=1, stride_y=1, padding='SAME')

        # Layer 4.
        # [Input] ==> 13x13x384
        # --> 13x13x384 ==> [Convolution: size=(3x3x384)x384, strides=1, padding=same] ==> 13x13x384
        # --> 13x13x384 ==> [ReLU] ==> 13x13x384
        # --> [Output] ==> 13x13x384
        with tf.name_scope('layer4'):
            layer4_activations = self.__conv(input=layer3_activations, filter_width=3, filter_height=3,
                                             filters_count=384, stride_x=1, stride_y=1, padding='SAME')

        # Layer 5.
        # [Input] ==> 13x13x384
        # --> 13x13x384 ==> [Convolution: size=(3x3x384)x256, strides=1, padding=same] ==> 13x13x256
        # --> 13x13x256 ==> [ReLU] ==> 13x13x256
        # --> 13x13x256 ==> [Max-Pool: size=3x3, strides=2, padding=valid] ==> 6x6x256
        # --> [Output] ==> 6x6x256
        with tf.name_scope('layer5'):
            layer5_activations = self.__conv(input=layer4_activations, filter_width=3, filter_height=3,
                                             filters_count=256, stride_x=1, stride_y=1, padding='SAME')
            layer5_pool = self.__max_pool(input=layer5_activations, filter_width=3, filter_height=3, stride_x=2,
                                          stride_y=2, padding='VALID')

        # Layer 6.
        # [Input] ==> 6x6x256=9216
        # --> 9216 ==> [Fully Connected: neurons=4096] ==> 4096
        # --> 4096 ==> [ReLU] ==> 4096
        # --> 4096 ==> [Dropout] ==> 4096
        # --> [Output] ==> 4096
        with tf.name_scope('layer6'):
            pool5_shape = layer5_pool.get_shape().as_list()
            flattened_input_size = pool5_shape[1] * pool5_shape[2] * pool5_shape[3]
            layer6_fc = self.__fully_connected(input=tf.reshape(layer5_pool, shape=[-1, flattened_input_size]),
                                               inputs_count=flattened_input_size, outputs_count=4096, relu=True)
            layer6_dropout = self.__dropout(input=layer6_fc)

        # Layer 7.
        # [Input] ==> 4096
        # --> 4096 ==> [Fully Connected: neurons=4096] ==> 4096
        # --> 4096 ==> [ReLU] ==> 4096
        # --> 4096 ==> [Dropout] ==> 4096
        # --> [Output] ==> 4096
        with tf.name_scope('layer7'):
            layer7_fc = self.__fully_connected(input=layer6_dropout, inputs_count=4096, outputs_count=4096, relu=True)
            layer7_dropout = self.__dropout(input=layer7_fc)

        # Layer 8.
        # [Input] ==> 4096
        # --> 4096 ==> [Logits: neurons=1000] ==> 1000
        # --> [Output] ==> 1000
        with tf.name_scope('layer8'):
            layer8_logits = self.__fully_connected(input=layer7_dropout, inputs_count=4096,
                                                   outputs_count=self.num_classes, relu=False, name='logits')

        # Cross Entropy.
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer8_logits, labels=self.Y,
                                                                       name='cross_entropy')
            self.__variable_summaries(cross_entropy)

        # Training.
        with tf.name_scope('training'):
            loss_operation = tf.reduce_mean(cross_entropy, name='loss_operation')
            tf.summary.scalar(name='loss', tensor=loss_operation)

            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum)

            # self.training_operation = optimizer.minimize(loss_operation, name='training_operation')

            grads_and_vars = optimizer.compute_gradients(loss_operation)
            self.training_operation = optimizer.apply_gradients(grads_and_vars, name='training_operation')

            for grad, var in grads_and_vars:
                if grad is not None:
                    with tf.name_scope(var.op.name + '/gradients'):
                        self.__variable_summaries(grad)

        # Accuracy.
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(layer8_logits, 1), tf.argmax(self.Y, 1), name='correct_prediction')
            self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_operation')
            tf.summary.scalar(name='accuracy', tensor=self.accuracy_operation)

    def train_epoch(self, sess, X_data, Y_data, batch_size=128, file_writer=None, summary_operation=None,
                    epoch_number=None):
        num_examples = len(X_data)
        step = 0
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_data[offset:end], Y_data[offset:end]
            if file_writer is not None and summary_operation is not None:
                _, summary = sess.run([self.training_operation, summary_operation],
                                      feed_dict={self.X: batch_x, self.Y: batch_y,
                                                 self.dropout_keep_prob: self.keep_prob})
                file_writer.add_summary(summary, epoch_number * (num_examples // batch_size + 1) + step)
                step += 1
            else:
                sess.run(self.training_operation, feed_dict={self.X: batch_x, self.Y: batch_y,
                                                             self.dropout_keep_prob: self.keep_prob})

    def evaluate(self, sess, X_data, Y_data, batch_size=128):
        num_examples = len(X_data)
        total_accuracy = 0
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_data[offset:end], Y_data[offset:end]
            batch_accuracy = sess.run(self.accuracy_operation, feed_dict={self.X: batch_x, self.Y: batch_y,
                                                                          self.dropout_keep_prob: 1.0})
            total_accuracy += (batch_accuracy * len(batch_x))
        return total_accuracy / num_examples

    def save(self, sess, file_name):
        saver = tf.train.Saver()
        saver.save(sess, file_name)

    def restore(self, sess, checkpoint_dir):
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

    def __variable_summaries(self, var):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.histogram('histogram', var)

    def __initial_weight_values(self, shape):
        # return tf.constant(value=0.01, dtype=tf.float32, shape=shape)
        return tf.random_normal(shape=shape, mean=self.random_mean, stddev=self.random_stddev, dtype=tf.float32)
    
    def __conv(self, input, filter_width, filter_height, filters_count, stride_x, stride_y, padding='VALID',
               name='conv'):
        # From article: All weights are initialized to 10**−2 and biases are set to 0.
        with tf.name_scope(name):
            input_channels = input.get_shape()[-1].value
            filters = tf.Variable(
                self.__initial_weight_values(shape=[filter_height, filter_width, input_channels, filters_count]),
                name='filters')
            convs = tf.nn.conv2d(input=input, filter=filters, strides=[1, stride_y, stride_x, 1], padding=padding,
                                 name='convs')
            biases = tf.Variable(tf.zeros(shape=[filters_count], dtype=tf.float32), name='biases')
            preactivations = tf.nn.bias_add(convs, biases, name='preactivations')
            activations = tf.nn.relu(preactivations, name='activations')

            with tf.name_scope('filter_summaries'):
                self.__variable_summaries(filters)

            with tf.name_scope('bias_summaries'):
                self.__variable_summaries(biases)

            with tf.name_scope('preactivations_histogram'):
                tf.summary.histogram('preactivations', preactivations)

            with tf.name_scope('activations_histogram'):
                tf.summary.histogram('activations', activations)

            return activations

    def __max_pool(self, input, filter_width, filter_height, stride_x, stride_y, padding='VALID', name='pool'):
        with tf.name_scope(name):
            pool = tf.nn.max_pool(input, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1],
                                  padding=padding, name='pool')
            return pool

    def __fully_connected(self, input, inputs_count, outputs_count, relu=True, name='fully_connected'):
        with tf.name_scope(name):
            wights = tf.Variable(self.__initial_weight_values(shape=[inputs_count, outputs_count]), name='wights')
            biases = tf.Variable(tf.zeros(shape=[outputs_count], dtype=tf.float32), name='biases')
            preactivations = tf.nn.bias_add(tf.matmul(input, wights), biases, name='preactivations')
            if relu:
                activations = tf.nn.relu(preactivations, name='activations')

            with tf.name_scope('wight_summaries'):
                self.__variable_summaries(wights)

            with tf.name_scope('bias_summaries'):
                self.__variable_summaries(biases)

            with tf.name_scope('preactivations_histogram'):
                tf.summary.histogram('preactivations', preactivations)

            if relu:
                with tf.name_scope('activations_histogram'):
                    tf.summary.histogram('activations', activations)

            if relu:
                return activations
            else:
                return preactivations

    def __dropout(self, input, name='dropout'):
        with tf.name_scope(name):
            return tf.nn.dropout(input, keep_prob=self.dropout_keep_prob, name='dropout')
