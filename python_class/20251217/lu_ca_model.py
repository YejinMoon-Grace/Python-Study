import tensorflow as tf
import config as cfg
import ce_an_utils as utils
from datetime import datetime
import lu_ca_tf_utils as tf_utils



class Model():
    def __init__(self, batch_size):
        ## Input
        # (batch, depth, height, width, channels)
        self.kernel_size = 3
        self.b_size = batch_size
        # self.X = tf.placeholder(tf.float32, shape=(batch_size, 64, 64, 1))
        # self.Y = tf.placeholder(tf.int32, shape=batch_size)
        self.X = tf.placeholder(tf.float32, shape=[self.b_size, 64, 64, 1])
        self.Y = tf.placeholder(tf.int32, shape=batch_size)

        self.train = True

        self.utf = tf_utils.utils(input_size=[1, 512, 512, 512, 1])

        self.loss, self.logit, self.pred = self.ResidualCNN()

    def ResidualCNN(self):

        conv_layer_shape = [[self.kernel_size, self.kernel_size, 1, 64],   # conv 64

                            [self.kernel_size, self.kernel_size, 64, 64],  # conv
                            [self.kernel_size, self.kernel_size, 64, 64],  # conv 64

                            [self.kernel_size, self.kernel_size, 64, 64],  # conv
                            [self.kernel_size, self.kernel_size, 64, 64],  # conv 64

                            [self.kernel_size, self.kernel_size, 64, 64],  # conv
                            [self.kernel_size, self.kernel_size, 64, 64],  # conv
                            # ---                  pool           ----#
                            [self.kernel_size, self.kernel_size, 64, 128],   # conv 32

                            [self.kernel_size, self.kernel_size, 128, 128],  # conv
                            [self.kernel_size, self.kernel_size, 128, 128],  # conv

                            [self.kernel_size, self.kernel_size, 128, 128],  # conv
                            [self.kernel_size, self.kernel_size, 128, 128],  # conv

                            [self.kernel_size, self.kernel_size, 128, 128],  # conv
                            [self.kernel_size, self.kernel_size, 128, 128],  # conv
                            # ---                  pool           ----#
                            [self.kernel_size, self.kernel_size, 128, 256],  # conv 16

                            [self.kernel_size, self.kernel_size, 256, 256],  # conv
                            [self.kernel_size, self.kernel_size, 256, 256],  # conv

                            [self.kernel_size, self.kernel_size, 256, 256],  # conv
                            [self.kernel_size, self.kernel_size, 256, 256],  # conv

                            [self.kernel_size, self.kernel_size, 256, 256],  # conv
                            [self.kernel_size, self.kernel_size, 256, 256],  # conv
                            # ---                  pool           ----#
                            [self.kernel_size, self.kernel_size, 256, 512],   # conv 8

                            [self.kernel_size, self.kernel_size, 512, 512],   # conv
                            [self.kernel_size, self.kernel_size, 512, 512],  # conv

                            [self.kernel_size, self.kernel_size, 512, 512],  # conv
                            [self.kernel_size, self.kernel_size, 512, 512],  # conv

                            [self.kernel_size, self.kernel_size, 512, 512],  # conv
                            [self.kernel_size, self.kernel_size, 512, 512],  # conv
                            # ---                  pool           ----#
                            [self.kernel_size, self.kernel_size, 512, 1024],   # conv 4
                            [self.kernel_size, self.kernel_size, 1024, 1024]]  # conv

        fc_layer_shape = [[4 * 4 * 1024, 1024], [1024, 1024], [1024, 2]]

        batch_size = self.b_size
        if self.train:
            drop_rate = 0.5
        else:
            drop_rate = 1.
        #     batch_size = self.b_size
        train_data_node = self.X
        train_labels_node = self.Y #

        # else:
        #     batch_size = 1
        # train_data_node = tf.placeholder(tf.float32, shape=(batch_size, 64, 64, 1))
        # train_labels_node = None

        cw = []
        cb = []

        fw = []
        fb = []
        layers = [train_data_node]

        # weight initialization
        cross_entropy, softmax = None, None
        for kernel, layer_cnt in zip(conv_layer_shape, range(len(conv_layer_shape))):
            w, b = self.utf.init_weight_bias(name="c%d" % (layer_cnt), shape=kernel,
                                             filtercnt=kernel[-1], trainable=True)
            cw.append(w)
            cb.append(b)

        for kernel, layer_cnt in zip(fc_layer_shape, range(len(fc_layer_shape))):
            w, b = self.utf.init_weight_bias(name="f%d" % (layer_cnt), shape=kernel, filtercnt=kernel[-1],
                                           trainable=True)
            fw.append(w)
            fb.append(b)

        # connect graph (stack layer)
        for w, b, layer_cnt in zip(cw, cb, range(len(cw))):

            if layer_cnt == 0 or layer_cnt == 7 or layer_cnt == 14 or layer_cnt == 21 or layer_cnt == 28:
                output = self.utf.conv_layer(data=layers[-1], weight=w, bias=b, padding="SAME")
                output = self.utf.relu_layer(output)
                layers.append(output)
                res_node = layers[-1]

            elif layer_cnt == 1 or layer_cnt == 8 or layer_cnt == 15 or layer_cnt == 22 or layer_cnt == 29:
                output = self.utf.conv_layer(data=layers[-1], weight=w, bias=b, padding="SAME")
                output = self.utf.relu_layer(output)
                layers.append(output)
            elif layer_cnt == 2 or layer_cnt == 9 or layer_cnt == 16 or layer_cnt == 23:
                output = self.utf.conv_layer(data=layers[-1], weight=w, bias=b, padding="SAME")
                output = tf.add(output, res_node)
                output = self.utf.relu_layer(output)
                layers.append(output)
            elif layer_cnt == 3 or layer_cnt == 10 or layer_cnt == 17 or layer_cnt == 24:
                res_node = layers[-1]
                output = self.utf.conv_layer(data=layers[-1], weight=w, bias=b, padding="SAME")
                output = self.utf.relu_layer(output)
                layers.append(output)
            elif layer_cnt == 4 or layer_cnt == 11 or layer_cnt == 18 or layer_cnt == 25:
                output = self.utf.conv_layer(data=layers[-1], weight=w, bias=b, padding="SAME")
                output = tf.add(output, res_node)
                output = self.utf.relu_layer(output)
                layers.append(output)
            elif layer_cnt == 5 or layer_cnt == 12 or layer_cnt == 19 or layer_cnt == 26:
                res_node = layers[-1]
                output = self.utf.conv_layer(data=layers[-1], weight=w, bias=b, padding="SAME")
                output = self.utf.relu_layer(output)
                layers.append(output)
            elif layer_cnt == 6 or layer_cnt == 13 or layer_cnt == 20 or layer_cnt == 27:
                output = self.utf.conv_layer(data=layers[-1], weight=w, bias=b, padding="SAME")
                output = tf.add(output, res_node)
                output = self.utf.relu_layer(output)
                layers.append(output)

                output = self.utf.pool_layer(data=layers[-1])
                layers.append(output)


        for w, b, layer_cnt in zip(fw, fb, range(len(fw))):
            if layer_cnt == 2:
                cross_entropy, softmax = self.utf.output_layer(data=layers[-1], weight=w, bias=b,
                                                                label=train_labels_node)
            else:
                output1 = self.utf.fc_layer_weight(data=layers[-1], weight=w, bias=b, dropout=drop_rate)
                layers.append(output1)

        predict = [tf.argmax(tf.cast(softmax > 0.6, tf.float32), 1), tf.argmax(tf.cast(softmax > 0.7, tf.float32), 1), tf.argmax(tf.cast(softmax > 0.8, tf.float32), 1)]

        return cross_entropy, softmax, predict

