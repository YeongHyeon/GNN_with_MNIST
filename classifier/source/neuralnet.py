import os
import tensorflow as tf
import source.layers as lay
import source.utils as utils

class Neuralnet(object):

    def __init__(self, height, width, ksize=3, learning_rate=1e-3, path_ckpt='', verbose=True):

        print("\nInitializing Neural Network...")
        self.height, self.width, self.ksize, self.learning_rate = \
            height, width, ksize, learning_rate
        self.path_ckpt = path_ckpt

        self.x = tf.compat.v1.placeholder(tf.float32, \
                shape=[None, self.height, self.width], name="x")
        self.batch_size = tf.compat.v1.placeholder(tf.int32, \
                shape=[], name="batch_size")
        self.training = tf.compat.v1.placeholder(tf.bool, shape=[], \
            name="training")

        self.layer = lay.Layers()
        self.variables, self.losses = {}, {}
        self.__build_model(x=self.x)
        self.__build_loss()

        print("\nNum Parameter: %d" %(self.layer.num_params))
        self.__init_session(path=self.path_ckpt)

    def step(self, x, iteration=0, training=False):

        feed_tr = {self.x:x, self.batch_size:x.shape[0], self.training:True}
        feed_te = {self.x:x, self.batch_size:x.shape[0], self.training:False}

        summary_list = []
        if(training):
            try:
                _, summaries = self.sess.run([self.optimizer, self.summaries], \
                    feed_dict=feed_tr, options=self.run_options, run_metadata=self.run_metadata)
                summary_list.append(summaries)
            except:
                _, summaries = self.sess.run([self.optimizer, self.summaries], \
                    feed_dict=feed_tr)
                summary_list.append(summaries)

            for summaries in summary_list:
                self.summary_writer.add_summary(summaries, iteration)

        loss, y_hat = \
            self.sess.run([self.losses['mse'], self.y_hat], feed_dict=feed_te)

        outputs = {'loss':loss, 'y_hat':y_hat}
        return outputs

    def save_parameter(self, model='model_checker'):

        self.saver.save(self.sess, os.path.join(self.path_ckpt, model))
        tf.io.write_graph(self.sess.graph_def, self.path_ckpt, "%s.pb" %(model), as_text=False)
        tf.io.write_graph(self.sess.graph_def, self.path_ckpt, "%s.pbtxt" %(model), as_text=True)

    def load_parameter(self, model='model_checker'):

        path_load = os.path.join(self.path_ckpt, '%s.index' %(model))
        if(os.path.exists(path_load)):
            print("\nRestoring parameters")
            self.saver.restore(self.sess, path_load.replace('.index', ''))

    def confirm_params(self, savedir='.', verbose=True):

        print("\n* Parameter arrange")

        ftxt = open(os.path.join(savedir, "list_parameters.txt"), "w")
        for var in tf.compat.v1.trainable_variables():
            text = "Trainable: " + str(var.name) + str(var.shape)
            if(verbose): print(text)
            ftxt.write("%s\n" %(text))
        ftxt.close()

    def loss_l1(self, x, reduce=None):

        distance = tf.compat.v1.reduce_mean(\
            tf.math.abs(x), axis=reduce)

        return distance

    def loss_l2(self, x, reduce=None):

        distance = tf.compat.v1.reduce_mean(\
            tf.math.sqrt(\
            tf.math.square(x) + 1e-9), axis=reduce)

        return distance

    def __init_session(self, path):
        try:
            print("\n* Initializing Session")
            sess_config = tf.compat.v1.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(config=sess_config)

            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.saver = tf.compat.v1.train.Saver()

            self.summary_writer = tf.compat.v1.summary.FileWriter(path, self.sess.graph)
            self.run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            self.run_metadata = tf.compat.v1.RunMetadata()
        except: pass

        flops = tf.profiler.profile(self.sess.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        print('FLOPS: %d' %(flops.total_float_ops))

    def __build_loss(self):

        self.losses['mse_b'] = self.loss_l2(self.y_hat - self.x, reduce=(1, 2))
        self.losses['mse'] = tf.compat.v1.reduce_mean(self.losses['mse_b'])

        tf.compat.v1.summary.scalar('NCAE/loss', self.losses['mse'])
        self.summaries = tf.compat.v1.summary.merge_all()

        self.variables['params'] = []
        for var in tf.compat.v1.trainable_variables():
            self.variables['params'].append(var)

        self.variables['ops'] = []
        for ops in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS):
            self.variables['ops'].append(ops)

        with tf.control_dependencies(self.variables['ops']):
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(\
                learning_rate=self.learning_rate).minimize(\
                loss=self.losses['mse'], var_list=self.variables['params'])

    def __build_model(self, x, verbose=True):

        self.variables['y_hat'] = self.auto_encoder(x=x, verbose=verbose)
        self.y_hat = tf.add(self.variables['y_hat'], 0, name="y_hat")

    def auto_encoder(self, x, name='ncae', verbose=True):

        for idx in range(3):
            x = self.layer.conv1d(x=x, stride=1, \
                filter_size=[self.ksize, self.width, self.width], dilations=[1, 1, 1], \
                padding='SAME', batch_norm=False, training=self.training, \
                activation='tanh', name="%s-%d" %(name, idx), verbose=verbose)

        return x
