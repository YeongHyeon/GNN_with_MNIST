import os
import tensorflow as tf
import source.layers as lay
import source.utils as utils

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

class Agent(object):

    def __init__(self, dim_n, dim_f, num_class, ksize=3, learning_rate=1e-3, path_ckpt='', verbose=True):

        print("\nInitializing Neural Network...")
        self.dim_n, self.dim_f, self.num_class, self.ksize, self.learning_rate = \
            dim_n, dim_f, num_class, ksize, learning_rate
        self.path_ckpt = path_ckpt

        self.variables = {}

        self.__model = Neuralnet(dim_n=self.dim_n, dim_f=self.dim_f, num_class=self.num_class, ksize=self.ksize)
        self.__model.forward(\
            x=tf.zeros((1, self.dim_n, self.dim_f), dtype=tf.float32), \
            a=tf.zeros((1, self.dim_n, self.dim_n), dtype=tf.float32), verbose=True)
        print("\nNum Parameter: %d" %(self.__model.layer.num_params))

        self.__init_propagation(path=self.path_ckpt)

    def __init_propagation(self, path):

        self.summary_writer = tf.summary.create_file_writer(self.path_ckpt)

        self.variables['trainable'] = []
        ftxt = open("list_parameters.txt", "w")
        for key in list(self.__model.layer.parameters.keys()):
            trainable = self.__model.layer.parameters[key].trainable
            text = "T: " + str(key) + str(self.__model.layer.parameters[key].shape)
            if(trainable):
                self.variables['trainable'].append(self.__model.layer.parameters[key])
            ftxt.write("%s\n" %(text))
        ftxt.close()

        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.save_params()

        conc_func = self.__model.__call__.get_concrete_function(\
            tf.TensorSpec(shape=(1, self.dim_n, self.dim_f), dtype=tf.float32), \
            tf.TensorSpec(shape=(1, self.dim_n, self.dim_n), dtype=tf.float32))

    def __loss(self, y, y_hat):

        entropy_b = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
        entropy = tf.math.reduce_mean(entropy_b)

        return {'entropy_b': entropy_b, 'entropy': entropy}

    @tf.autograph.experimental.do_not_convert
    def step(self, x, a, y, iteration=0, training=False):

        with tf.GradientTape() as tape:
            y_hat = self.__model.forward(x=x, a=a, verbose=False)
            losses = self.__loss(y=y, y_hat=y_hat)

        if(training):
            gradients = tape.gradient(losses['entropy'], self.variables['trainable'])
            self.optimizer.apply_gradients(zip(gradients, self.variables['trainable']))

            with self.summary_writer.as_default():
                tf.summary.scalar('%s/entropy' %(self.__model.who_am_i), losses['entropy'], step=iteration)

        return {'y_hat':y_hat, 'losses':losses}

    def save_params(self, model='base'):

        vars_to_save = self.__model.layer.parameters.copy()
        vars_to_save["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_save)
        ckptman = tf.train.CheckpointManager(ckpt, directory=os.path.join(self.path_ckpt, model), max_to_keep=1)
        ckptman.save()

    def load_params(self, model):

        vars_to_load = self.__model.layer.parameters.copy()
        vars_to_load["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_load)
        latest_ckpt = tf.train.latest_checkpoint(os.path.join(self.path_ckpt, model))
        status = ckpt.restore(latest_ckpt)
        status.expect_partial()

    def loss_l1(self, x, reduce=None):

        distance = tf.math.reduce_mean(\
            tf.math.abs(x), axis=reduce)

        return distance

    def loss_l2(self, x, reduce=None):

        distance = tf.math.reduce_mean(\
            tf.math.sqrt(\
            tf.math.square(x) + 1e-9), axis=reduce)

        return distance

class Neuralnet(tf.Module):

    def __init__(self, dim_n, dim_f, num_class, ksize):
        super(Neuralnet, self).__init__()

        self.who_am_i = "GAT"
        self.dim_n, self.dim_f, self.num_class = dim_n, dim_f, num_class

        self.layer = lay.Layers()

        self.forward = tf.function(self.__call__)

    @tf.function
    def __call__(self, x, a, verbose=False):

        y_hat = self.__classifier(x=x, a=a, name='clf', verbose=verbose)
        y_hat = tf.add(y_hat, 0, name="y_hat") # speeds up training trick

        return y_hat

    def __classifier(self, x, a, name='clf', verbose=True):

        if(verbose): print("\n* Classifier")

        # Step 1. Graph Convolutional Layers
        a = self.layer.attntion_coeffcient(x=x, a=a, name='%s_at' %(name), verbose=verbose)
        for idx in range(3):
            x = self.layer.graph_conv(x=x, a=a, c_out=16*(2**idx), \
                batch_norm=False, activation='relu', name='%s-%d' %(name, idx), verbose=verbose)

        # Step 2. Readout Layer
        x = self.layer.read_out(x=x, c_out=16*(2**idx+1), \
            batch_norm=False, activation='elu', name='%s-readout' %(name), verbose=verbose)

        # Step 3. Classifier
        x = self.layer.fully_connected(x=x, c_out=256, \
                batch_norm=False, activation='relu', name="%s-clf0" %(name), verbose=verbose)
        x = self.layer.fully_connected(x=x, c_out=self.num_class, \
                batch_norm=False, activation=None, name="%s-clf1" %(name), verbose=verbose)

        return x
