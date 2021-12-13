import numpy as np
import tensorflow as tf

class Layers(object):

    def __init__(self, parameters={}):

        self.num_params = 0
        self.parameters = {}
        self.initializer_xavier = tf.initializers.glorot_normal()

    """ -*-*-*-*-*- Variables -*-*-*-*-*- """

    def __get_variable(self, shape, constant=None, trainable=True, name=''):

        try: return self.parameters[name]
        except:
            if(constant is None):
                w = tf.Variable(
                    initial_value=self.initializer_xavier(shape),
                    trainable=True,
                    dtype=tf.float32,
                    name="%s_w" %(name)
                )
            else:
                w = tf.constant(
                    value=constant,
                    shape=shape,
                    dtype=tf.float32,
                    name="%s_w" %(name)
                )

            tmp_num = 1
            for num in shape:
                tmp_num *= num
            self.num_params += tmp_num
            self.parameters[name] = w

            return self.parameters[name]

    """ -*-*-*-*-*- Classic Functions -*-*-*-*-*- """
    def activation(self, x, activation=None, name=''):

        if(activation is None): return x
        elif("sigmoid" == activation):
            return tf.nn.sigmoid(x, name='%s_sigmoid' %(name))
        elif("tanh" == activation):
            return tf.nn.tanh(x, name='%s_tanh' %(name))
        elif("relu" == activation):
            return tf.nn.relu(x, name='%s_relu' %(name))
        elif("lrelu" == activation):
            return tf.nn.leaky_relu(x, name='%s_lrelu' %(name))
        elif("elu" == activation):
            return tf.nn.elu(x, name='%s_elu' %(name))
        elif("swish" == activation):
            return tf.nn.swish(x, name='%s_swish' %(name))
        else: return x

    def batch_normalization(self, x, trainable=True, name='', verbose=True):

        x_mean, x_var = tf.nn.moments(x, keepdims=False)

        y = tf.nn.batch_normalization(
            x=x,
            mean=x_mean,
            variance=x_var,
            name=name
        )

        if(verbose): print("BN (%s)" %(name), x.shape, ">", y.shape)
        return y

    def conv1d(self, x, stride, \
        filter_size=[3, 16, 32], dilations=[1, 1, 1], \
        padding='SAME', batch_norm=False, activation=None, name='', verbose=True):

        w = self.__get_variable(shape=filter_size, \
            trainable=True, name='%s_w' %(name))
        b = self.__get_variable(shape=[filter_size[-1]], \
            trainable=True, name='%s_b' %(name))

        wx = tf.nn.conv1d(
            input=x,
            filters=w,
            stride=stride,
            padding=padding,
            data_format='NWC',
            dilations=None,
            name='%s_conv' %(name)
        )

        y = tf.math.add(wx, b, name='%s_add' %(name))
        if(verbose): print("Conv (%s)" %(name), x.shape, "->", y.shape)

        if(batch_norm): y = self.batch_normalization(x=y, \
            trainable=True, name='%s_bn' %(name), verbose=verbose)
        return self.activation(x=y, activation=activation, name=name)

    def convt1d(self, x, stride, output_shape, \
        filter_size=[3, 16, 32], dilations=[1, 1, 1], \
        padding='SAME', batch_norm=False, activation=None, name='', verbose=True):

        for idx_os, _ in enumerate(output_shape):
            if(idx_os == 0): continue
            output_shape[idx_os] = int(output_shape[idx_os])

        w = self.__get_variable(shape=filter_size, \
            trainable=True, name='%s_w' %(name))
        b = self.__get_variable(shape=[filter_size[-2]], \
            trainable=True, name='%s_b' %(name))

        wx = tf.nn.conv1d_transpose(
            input=x,
            filters=w,
            output_shape=output_shape,
            strides=stride,
            padding=padding,
            data_format='NWC',
            dilations=dilations,
            name='%s_conv_tr' %(name)
        )

        y = tf.math.add(wx, b, name='%s_add' %(name))
        if(verbose): print("ConvT (%s)" %(name), x.shape, "->", y.shape)

        if(batch_norm): y = self.batch_normalization(x=y, \
            trainable=True, name='%s_bn' %(name), verbose=verbose)
        return self.activation(x=y, activation=activation, name=name)

    def conv2d(self, x, stride, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], \
        padding='SAME', batch_norm=False, activation=None, name='', verbose=True):

        w = self.__get_variable(shape=filter_size, \
            trainable=True, name='%s_w' %(name))
        b = self.__get_variable(shape=[filter_size[-1]], \
            trainable=True, name='%s_b' %(name))

        wx = tf.nn.conv2d(
            input=x,
            filters=w,
            strides=[1, stride, stride, 1],
            padding=padding,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv' %(name)
        )

        y = tf.math.add(wx, b, name='%s_add' %(name))
        if(verbose): print("Conv (%s)" %(name), x.shape, "->", y.shape)

        if(batch_norm): y = self.batch_normalization(x=y, \
            trainable=True, name='%s_bn' %(name), verbose=verbose)
        return self.activation(x=y, activation=activation, name=name)

    def convt2d(self, x, stride, output_shape, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], \
        padding='SAME', batch_norm=False, activation=None, name='', verbose=True):

        for idx_os, _ in enumerate(output_shape):
            if(idx_os == 0): continue
            output_shape[idx_os] = int(output_shape[idx_os])

        w = self.__get_variable(shape=filter_size, \
            trainable=True, name='%s_w' %(name))
        b = self.__get_variable(shape=[filter_size[-2]], \
            trainable=True, name='%s_b' %(name))

        wx = tf.nn.conv2d_transpose(
            input=x,
            filters=w,
            output_shape=output_shape,
            strides=[1, stride, stride, 1],
            padding=padding,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv_tr' %(name)
        )

        y = tf.math.add(wx, b, name='%s_add' %(name))
        if(verbose): print("ConvT (%s)" %(name), x.shape, "->", y.shape)

        if(batch_norm): y = self.batch_normalization(x=y, \
            trainable=True, name='%s_bn' %(name), verbose=verbose)
        return self.activation(x=y, activation=activation, name=name)

    def fully_connected(self, x, c_out, \
        batch_norm=False, activation=None, name='', verbose=True):

        c_in, c_out = x.get_shape().as_list()[-1], int(c_out)

        w = self.__get_variable(shape=[c_in, c_out], \
            trainable=True, name='%s_w' %(name))
        b = self.__get_variable(shape=[c_out], \
            trainable=True, name='%s_b' %(name))

        wx = tf.linalg.matmul(x, w, name='%s_mul' %(name))
        y = tf.math.add(wx, b, name='%s_add' %(name))
        if(verbose): print("FC (%s)" %(name), x.shape, "->", y.shape)

        if(batch_norm): y = self.batch_normalization(x=y, \
            trainable=True, name='%s_bn' %(name), verbose=verbose)
        return self.activation(x=y, activation=activation, name=name)

    """ -*-*-*-*-*- Custom Functions -*-*-*-*-*- """
    def trim_odd(self, x):

        b, t, c = list(x.get_shape())
        if(int(t) % 2 == 1):
            return tf.slice(x, [0, 0, 0], [-1, int(t)-1, -1])
        else:
            return x

    def trim_shape(self, x, shape):

        return tf.slice(x, [0, 0, 0], shape)

    def sub_pixel1d(self, x, ratio, verbose=True):

        y1 = tf.transpose(a=x, perm=[2, 1, 0]) # (r, w, b)
        y2 = tf.batch_to_space(y1, [ratio], [[0, 0]]) # (1, r*w, b)
        y3 = tf.transpose(a=y2, perm=[2, 1, 0])

        if(verbose): print("SubPixel", x.shape, "->", y3.shape)
        return y3

    def graph_conv(self, x, a, c_out, \
        batch_norm=False, activation=None, name='', verbose=True):

        c_in, c_out = x.get_shape().as_list()[-1], int(c_out)

        w = self.__get_variable(shape=[c_in, c_out], \
            trainable=True, name='%s_w' %(name))
        b = self.__get_variable(shape=[c_out], \
            trainable=True, name='%s_b' %(name))

        wx = tf.linalg.matmul(x, w, name='%s_mul' %(name))
        y_feat = tf.math.add(wx, b, name='%s_add' %(name))
        y = tf.linalg.matmul(a, y_feat)

        if(verbose): print("G-Conv (%s)" %(name), x.shape, "->", y.shape)

        if(batch_norm): y = self.batch_normalization(x=y, \
            trainable=True, name='%s_bn' %(name), verbose=verbose)
        return self.activation(x=y, activation=activation, name=name)

    def attntion_coeffcient(self, x, a, name='', verbose=True):

        num_node, c_in = x.get_shape().as_list()[-2], x.get_shape().as_list()[-1]
        w = self.__get_variable(shape=[c_in, c_in], \
            trainable=True, name='%s_w' %(name))

        wx = tf.linalg.matmul(x, w, name='%s_mul' %(name))
        wxi = tf.repeat(wx, repeats=x.shape[1], axis=1)
        wxj = tf.tile(wx, multiples=[1, x.shape[1], 1])

        eij = tf.reshape(tf.concat([wxi, wxj], axis=1), (-1, num_node, num_node, 2*c_in))
        avg_k = tf.reduce_mean(eij, axis=-1)
        y = tf.nn.softmax(avg_k, axis=-1)

        if(verbose): print("G-attn (%s)" %(name), x.shape, a.shape, "->", y.shape)
        return y

    def read_out(self, x, c_out, \
        batch_norm=False, activation=None, name='', verbose=True):

        wx = self.fully_connected(x=x, c_out=c_out, \
            batch_norm=False, activation=None, name=name, verbose=False)

        y = tf.math.reduce_sum(wx, axis=-1)

        if(verbose): print("Readout (%s)" %(name), x.shape, "->", y.shape)
        return self.activation(x=y, activation=activation, name=name)
