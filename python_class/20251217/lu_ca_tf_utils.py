
import math
import numpy as np
import tensorflow as tf


class utils:
    def __init__(self, input_size):
        self.data_size = input_size
        self.alpha = 0.1
        self.initializer = tf.contrib.layers.variance_scaling_initializer()  # HE initializer.
        self.regularizer = None  # tf.contrib.layers.l2_regularizer(0.00001)


    def init_weight_bias(self, name, shape, filtercnt, trainable):
        weights = tf.get_variable(name=name + "w", shape=shape,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                  dtype=tf.float32, trainable=trainable)
        biases = tf.Variable(initial_value=tf.constant(0.1, shape=[filtercnt], dtype=tf.float32), name=name + "b",
                             trainable=trainable)
        return weights, biases

    def init_weight_bias_3d(self, name, shape, filtercnt, trainable):
        weights = tf.get_variable(name=name + "w", shape=shape,
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  dtype=tf.float32, trainable=trainable)
        biases = tf.Variable(initial_value=tf.constant(0.1, shape=[filtercnt], dtype=tf.float32), name=name + "b",
                             trainable=trainable)
        return weights, biases
    def layer_conv2D(self, name, inputs, filters, kernel_size, strides, padding='valid'):
        """
        2D convolution using tf.layers
        There are different 2D convolution functions at tensorflow like tf.nn.conv2d, tf.layers.conv2d, tf.contrib.layers.conv2d.
        Args:
            name: Layer name. string type.
            inputs: Input data. tensor type.
            filters: number of output channel. int type.
            kernel_size: list of filter size. list or tuple type. ex) if you want to use 3x3 filter, filters should be [3, 3]
            strides: number of stride. int type.
            padding: string of padding option. Options are 'valid' and 'same'. string type.

        Returns: convolution results, tensor type.

        """
        conv2D = tf.layers.conv2d(inputs=inputs, filters=filters,
                                  kernel_size=kernel_size, strides=strides,
                                  padding=padding, use_bias=True,
                                  kernel_initializer=self.initializer,
                                  kernel_regularizer=self.regularizer,
                                  name=name)
        return conv2D

    def layer_conv3D(self, name, inputs, filters, kernel_size, strides, padding='valid'):
        conv3D = tf.layers.conv3d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                  padding=padding, use_bias=True, kernel_initializer=self.initializer,
                                  kernel_regularizer=self.regularizer, name=name)
        return conv3D

    def layer_s_conv2D(self,name, inputs, filters, kernel_size, strides, padding='valid'):
        """
        2D seperable convolution using tf.layers
        Also, there are different 2D separable convolution at tensorflow.
        Args:
            name: Layer name. string type.
            inputs: Input data. tensor type.
            filters: number of output channel. int type.
            kernel_size: list of filter size. list or tuple type. ex) if you want to use 3x3 filter, filters should be [3, 3]
            strides: number of stride. int type.
            padding: string of padding option. Options are 'valid' and 'same'. string type.

        Returns: seperable convolution results, tensor type.

        """
        s_conv2D = tf.layers.separable_conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                              padding=padding, use_bias=True,
                                              depthwise_initializer=self.initializer,
                                              depthwise_regularizer=self.regularizer,
                                              pointwise_initializer=self.initializer,
                                              pointwise_regularizer=self.regularizer,
                                              name=name)
        return s_conv2D

    def conv_layer(self, data, weight, bias, padding):
        conv = tf.nn.conv2d(input=data, filter=weight, strides=[1, 1, 1, 1], padding=padding)
        # return tf.nn.relu(tf.nn.bias_add(conv, bias))
        return tf.nn.bias_add(conv, bias)

    def squeeze_layer(self, output):
        return tf.squeeze(output, squeeze_dims=-1)

    def up_conv_layer(self, data, weight, bias, padding):

        shape_output = data.get_shape().as_list()
        shape_output[1] *= 2
        shape_output[2] *= 2
        shape_output[-1] = shape_output[-1] // 2

        upconv = tf.nn.conv2d_transpose(value=data, filter=weight, output_shape=shape_output,
                                        strides=[1, 2, 2, 1], padding=padding)

        return tf.nn.relu(tf.nn.bias_add(upconv, bias))

    def up_conv_3d_layer(self, data, weight, bias, stride, padding):

        shape_output = data.get_shape().as_list()
        shape_output[1] *= 2
        shape_output[2] *= 2
        shape_output[3] *= 2
        shape_output[-1] = bias.get_shape().as_list()[0]

        upconv = tf.nn.conv3d_transpose(value=data, filter=weight, output_shape=shape_output,
                                        strides=stride, padding=padding)

        return tf.nn.bias_add(upconv, bias)


    def crop_concat(self, data, conv):

        # crop and concat (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2])
        shape1 = conv.get_shape().as_list()
        shape2 = data.get_shape().as_list()
        offsets = [0, (shape1[1]-shape2[1]) // 2, (shape1[2]-shape2[2]) // 2, 0]
        size = [-1, shape2[1], shape2[2], -1]
        crop_conv = tf.slice(conv, offsets, size)

        return tf.concat([data, crop_conv], axis=3)

    def crop_concat_3d(self, data, conv):

        # crop and concat (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2])
        shape1 = conv.get_shape().as_list()
        shape2 = data.get_shape().as_list()
        offsets = [0, (shape1[1]-shape2[1]) // 2, (shape1[2]-shape2[2]) // 2, (shape1[3]-shape2[3]) // 2, 0]
        size = [-1, shape2[1], shape2[2], shape2[3], -1]
        crop_conv = tf.slice(conv, offsets, size)

        return tf.concat([data, crop_conv], axis=-1)

    def reshape_layer(self, conv):
        return tf.expand_dims(conv, -1)

    def conv3d_layer(self, data, weight, bias, stride, padding):
        conv = tf.nn.conv3d(input=data, filter=weight, strides=stride, padding=padding)
        return tf.nn.bias_add(conv, bias)

    def batch_norm_layer(self, data, train=True):
        return tf.contrib.layers.batch_norm(inputs=data, is_training=train)

    def parametric_relu_layer(self, conv, alpha):
        return alpha * conv if conv < 0 else tf.nn.relu(conv)

    def relu_layer(self, conv):
        return tf.nn.relu(conv)

    def leaky_relu_layer(self, conv, alpha):
        return tf.nn.relu(conv) - alpha * tf.nn.relu(-conv)

    def depth_wise_conv_layer(self, data, weight, bias, padding, is_inception):
        conv = tf.nn.depthwise_conv2d(input=data, filter=weight, strides=[1, 1, 1, 1], padding=padding)
        if is_inception:
            return tf.nn.bias_add(conv, bias)
        return tf.nn.relu(tf.nn.bias_add(conv, bias))


    def deconv2D(self, name, inputs, filter_shape, output_shape, strides, padding='valid'):
        """
        2D transpose convolution using tf.nn
        Also, there are different 2D transpose convolution at tensorflow.
        Some bugs are occured at tf.layers.conv2d_transpose, so manually using tf.nn.conv2d_tranpose.
        Weight(W) and shape, batchsize, output shape must be declared manually.
        Args:
            name: Layer name. string type.
            inputs: Input data. tensor type.
            filter_shape: list of filter shape, shape will be [filter size, filter size, output channel, input channel]. list of tuple type.
            output_shape: list of output shape, shape will be [batch_size, output size, output size, output channel], -1 for automatically fitting to batch size. list or tuple type.
            strides: list of stride shape, if you want to stride [2, 2], shape will be [1, 2, 2, 1]
            padding: string of padding option. Options are 'valid' and 'same'. string type.

        Returns: transpose convolution results, tensor type.
        """

        W = tf.get_variable(name + 'W', filter_shape, initializer=self.initializer, regularizer=self.regularizer)
        # shape = tf.shape(inputs)

        # output_shape2 = [batch_size, output_shape[1], output_shape[2], output_shape[3]]
        layer = tf.nn.conv2d_transpose(inputs, filter=W, output_shape=output_shape, strides=strides, padding=padding)
        return layer

    def re_conv2D(self, name, inputs, output_shape):
        """
        https://distill.pub/2016/deconv-checkerboard/
        re-convolution can replace transpose convolution. Used at Cycle GAN.
        Instead of transpose convolution, resize images with nearest neighbor interpolation, after then using 1x1 convolution for reshaping channels.
        Args:
            name: Layer name. string type.
            inputs: Input data. tensor type.
            output_shape: list of output shape, shape will be [-1, output size, output size, output channel], -1 for automatically fitting to batch size. list or tuple type.

        Returns: resize convolution results, tensor type.

        """
        resize_layer = tf.image.resize_nearest_neighbor(images=inputs, size=[output_shape[1], output_shape[2]],
                                                        name=name + '_resizing')
        # padding_layer = tf.pad(resize_layer)
        # conv_layer = conv2D(padding_layer)
        conv_layer = self.layer_conv2D(name=name + '_conv', inputs=resize_layer, filters=output_shape[3],
                                       kernel_size=[3, 3], strides=[1, 1], padding='same')
        return conv_layer

    def pool_layer(self, data, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='VALID'):
        # kernel = [1, 2, 2, 1] stride = [1, 2, 2, 1]
        return tf.nn.max_pool(value=data, ksize=kernel, strides=stride, padding=padding)

    def dense_layer(self, data, unit, drop, trainable):
        shape = data.get_shape().as_list()
        data_flat = tf.reshape(data, [shape[0], np.prod(shape[1:])])
        dropout = tf.layers.dropout(inputs=data_flat, rate=drop, training=trainable)
        dense = tf.layers.dense(inputs=dropout, units=unit, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=drop, training=trainable)
        dense = tf.layers.dense(inputs=dropout, units=np.prod(shape[1:]), activation=tf.nn.relu)

        return tf.reshape(dense, [shape[0], shape[1], shape[2], shape[3]])

    def fc_layer(self, data, dropout, trainable):
        shape = data.get_shape().as_list()
        shape = [np.prod(shape[1:]), shape[0]]

        w, b = self.init_weight_bias(name='fc', shape=shape, filtercnt=shape[-1], trainable=trainable)

        hidden = tf.nn.bias_add(tf.matmul(tf.reshape(data, [shape[1], shape[0]]), w), b)
        hidden = tf.nn.relu(hidden)
        if dropout < 1.:
            hidden = tf.nn.dropout(hidden, dropout)
        return hidden

    def fc_layer_weight(self, data, weight, bias, dropout):
        shape = data.get_shape().as_list()
        shape = [shape[0], np.prod(shape[1:])]

        hidden = tf.nn.bias_add(tf.matmul(tf.reshape(data, shape), weight), bias)

        hidden = tf.nn.relu(hidden)
        if dropout < 1.:
            hidden = tf.nn.dropout(hidden, dropout)
        return hidden

    def fc_concat_layer(self, data1, data2, weight, bias, dropout, batch_norm=False):
        data = tf.concat([data1, data2], axis=3)
        shape = data.get_shape().as_list()
        shape = [shape[0], np.prod(shape[1:])]

        hidden = tf.nn.bias_add(tf.matmul(tf.reshape(data, shape), weight), bias)
        if batch_norm:
            hidden = tf.contrib.layers.batch_norm(inputs=hidden, is_training=True)
            # self.batch_norm_layer(hidden)
        hidden = tf.nn.relu(hidden)
        if dropout < 1.:
            hidden = tf.nn.dropout(hidden, dropout)
        return hidden

    def huber_loss(self, labels, predictions, delta=1.0):
        residual = tf.abs(predictions - labels)
        condition = tf.less(residual, delta)
        small_res = 0.5 * tf.square(residual)
        large_res = delta * residual - 0.5 * tf.square(delta)
        return tf.where(condition, small_res, large_res)

    def conv_out_size_same(self, size, stride):
            return int(math.ceil(float(size) / float(stride)))

    def modified_focal_loss(self, output, target, total_weight=1., b_weight=0.5):
        """
        focal loss is loss function for detect rare, small class. This is modified focal loss, modified by LYE.
        Args:
            output: predicted, output of network
            target: groundtruth, label data
            total_weight: total weight
            b_weight: background weight

        Returns: loss value.

        """
        """
        기존에 사용하던 focal_loss는 foreground area(object label이 있는 area)의 오차만을 loss 값에 반영하였는데, 
        이렇게 하니 background area(object label이 없는 area)에서 마구잡이로 object가 있다고 찾아버리는 문제가 발생한다.
        이를 해결하기 위해 background area의 예측 오차도 loss 값에 반영하도록 하였다.

        :param output: 모델을 통해 나온 예측값
        :param target: 정답 라벨
        :param total_weight: 총 loss 값의 크기를 조절하는 가중치.
        :param b_weight: background loss의 반영 비율을 결정하는 가중치.
                가중치를 낮게 줄수록 foreground area를 민감하게 잡을 수 있고, 0으로 하면 foreground loss만 100% 반영할 수 있다.
        :return: 산출된 loss 값. total_weight * (foreground loss + (b_weight * background loss))
        """
        foreground_predicted, background_predicted = tf.split(output, [1, 1], 3)
        foreground_truth, background_truth = tf.split(target, [1, 1], 3)

        foreground_loss = self.focal_loss(output=foreground_predicted, target=foreground_truth)
        background_loss = self.focal_loss(output=background_predicted, target=background_truth)

        return total_weight * (((1 - b_weight) * foreground_loss) + (b_weight * background_loss))

    def focal_loss(self, output, target, smooth=1e-6):
        """
        Original focal loss. focal loss is loss function for detect rare, small class.
        Check https://arxiv.org/pdf/1708.02002.pdf, focal loss paper and https://arxiv.org/pdf/1711.01506.pdf, focal unet paper.
        Args:
            output: predicted, output of network
            target: groundtruth, label data
            smooth: very small value for avoid zero divide.

        Returns: loss value.

        """
        output, _ = tf.split(output, [1, 1], 3)
        target, _ = tf.split(target, [1, 1], 3)
        focal_matrix = -tf.square(tf.ones_like(output) - output) * target * tf.log(output + smooth)
        focal = tf.reduce_sum(focal_matrix)
        return focal

    # def focal_loss_backup(output, target, smooth=1e-6):
    #     focal = -tf.reduce_sum(tf.square(tf.ones_like(output) - output) * target * tf.log(output + smooth))
    #     return focal

    # '''
    # 문제 : label이 없는 경우 predict에서 픽셀을 단 하나만 집어도 로스가 매우 크게 적용된다.
    # 대안 : inse, l, r의 reduce_sum을 reduce_mean으로 수정
    # 1. pixel-wise로 각각 곱해준다
    # 2. 배치단위로 각각 평균을 내준다
    # 3. 배치별로 dice loss를 구한다
    # 4. 배치 전체를 평균낸다
    #
    # * 추가 대안
    # 1. 틀린 픽셀의 갯수에 비례해서 로그적으로 로스가 증가하게 한다
    # 2. 있는 걸 없다고 체크한 오답에 대해 더 큰 로스를 적용한다
    # '''
    def mean_square_loss(self, output, target):
        return tf.losses.mean_squared_error(labels=target, predictions=output)

    def modified_dice_loss(self, output, target, axis=(1, 2, 3), smooth=1e-6):
        """
        dice loss is loss function using as true positives/(true positives + false negatives + false positives). This is modified dice loss, modified by LYE.
        similar with IoU.
        Args:
            output: predicted, output of network
            target: groundtruth, label data
            axis: axis for reduce_sum. if 2D dataset like [batch, height, width, channel], axis will be [1, 2, 3] or (1, 2, 3)
            smooth: very small value for avoid zero divide.

        Returns: loss value

        """
        output, _ = tf.split(output, [1, 1], 3)
        target, _ = tf.split(target, [1, 1], 3)

        inse = tf.reduce_mean(output * target, axis=axis)
        l = tf.reduce_mean(output * output, axis=axis)
        r = tf.reduce_mean(target * target, axis=axis)
        dice = (2. * inse + smooth) / (l + r + smooth)
        dice = tf.reduce_mean(dice)
        return 1 - dice

    def dice_loss(self, output, target, axis=(1, 2, 3), smooth=1e-6):
        """
        dice loss is loss function using as true positives/(true positives + false negatives + false positives).
        similar with IoU.
        Args:
            output: predicted, output of network
            target: groundtruth, label data
            axis: axis for reduce_sum. if 2D dataset like [batch, height, width, channel], axis will be [1, 2, 3] or (1, 2, 3)
            smooth: very small value for avoid zero divide.

        Returns: loss value

        """
        output, _ = tf.split(output, [1, 1], 3)
        target, _ = tf.split(target, [1, 1], 3)

        inse = tf.reduce_sum(output * target, axis=axis)
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
        dice = (2. * inse + smooth) / (l + r + smooth)
        dice = tf.reduce_mean(dice)
        return 1 - dice

    def cross_entropy(self, output, target):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output))

    def softmax(self, output):
        return tf.nn.softmax(output)

    def output_layer(self, data, weight, bias, label):
        shape = data.get_shape().as_list()
        shape = [shape[0], np.prod(shape[1:])]
        hidden = tf.nn.bias_add(tf.matmul(tf.reshape(data, shape), weight), bias)

        if label is None:
            return None, tf.nn.softmax(hidden)
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=hidden,
                                                                             labels=label)), tf.nn.softmax(hidden)

    def pixel_wise_softmax(self, output_map):
        with tf.name_scope("pixel_wise_softmax"):
            max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
            exponential_map = tf.exp(output_map - max_axis)
            normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
            return exponential_map / normalize

    def pixel_wise_softmax_minsky(self, output_map):
        max_axis = tf.reduce_max(output_map, axis=3, keep_dims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keep_dims=True)
        return exponential_map / normalize

    def pixel_wise_cross_entropy(self, label, output_map):
        return -tf.reduce_sum(label * tf.log(tf.clip_by_value(output_map, 1e-10, 1.0)), name="cross_entropy")
        # logits = tf.log(tf.clip_by_value(output_map, 1e-10, 1.0))
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(label, output_map)

        # cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        # tf.add_to_collection('losses', cross_entropy_mean)

        # return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def dice_loss2(self, label, logit, smooth=1e-6, num_class=2):
        label_oh = tf.squeeze(tf.one_hot(label, num_class, dtype=tf.float32))
        logit_sm = tf.squeeze(tf.nn.softmax(logit, axis=-1))

        dice_lst = []

        for label_ch, logit_ch in zip(tf.split(label_oh, num_class, axis=-1), tf.split(logit_sm, num_class, axis=-1)):
            numerator = 2. * tf.reduce_sum(tf.multiply(label_ch, logit_ch))
            denominator = tf.reduce_sum(tf.square(label_ch)) + tf.reduce_sum(tf.square(logit_ch))

            dice = (numerator + smooth) / (denominator + smooth)
            dice_lst.append(dice)

        return 1 - tf.reduce_mean(dice_lst)

    def pixel_wise_ouput_layer(self, label, data, trainable):

        if trainable is False:
            return None, self.pixel_wise_softmax(data)
        else:
            shape1 = label.get_shape().as_list()
            shape2 = data.get_shape().as_list()
            if not shape1[1:2] == shape2[1:2]:
                offsets = [0, (shape1[1] - shape2[1]) // 2, (shape1[2] - shape2[2]) // 2, 0]
                size = [-1, shape2[1], shape2[2], -1]
                label = tf.slice(label, offsets, size)

            # return self.pixel_wise_cross_entropy(label, data), self.pixel_wise_softmax(data)
            return self.dice_loss2(label, data, num_class=2), self.pixel_wise_softmax(data)

    def input_layer(self, batch_size=128, train=True):
        if train:
            data_node = tf.placeholder(tf.float32,
                                       shape=(batch_size, self.data_size[0], self.data_size[1], self.data_size[2]))
            label_node = tf.placeholder(tf.int64, shape=batch_size)
        else:
            data_node = tf.placeholder(tf.float32,
                                       shape=(batch_size, self.data_size[0], self.data_size[1], self.data_size[2]))
            label_node = None

        return data_node, label_node
