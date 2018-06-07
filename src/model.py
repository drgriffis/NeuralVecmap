'''
Tensorflow implementation of feed-forward DNN to learn a
nonlinear projection from one set of embeddings to another,
based on common keys.

Model parameters include:
  - # of hidden layers
  - Activation function (tanh or ReLU)
  - Dimensionality of hidden layers

Model is trained with MSE loss between the current projections
and reference embeddings from the target set.
'''

import tensorflow as tf

class MapperParams:
    src_dim = 0
    trg_dim = 0
    map_dim = 0
    activation = 'tanh'
    num_layers = 1
    checkpoint_file = 'checkpoint'

    def __init__(self, **kwargs):
        for (k,v) in kwargs.items():
            self.__dict__[k] = v

class ManifoldMapper:
    
    def __init__(self, session, params, random_seed=None):
        self._session = session
        self.p = params
        self._build()
        if random_seed:
            tf.set_random_seed(random_seed)
        self._session.run([self._init])

        self.saver = tf.train.Saver()

    def checkpoint(self, cur_iter):
        self.saver.save(self._session, self.p.checkpoint_file, global_step=cur_iter)

    def rollback(self):
        self.saver.restore(self._session, self.saver.last_checkpoints[-1])

    def _build(self):
        
        self._src_embeddings = tf.placeholder(tf.float32, shape=[None, self.p.src_dim])
        self._trg_embeddings = tf.placeholder(tf.float32, shape=[None, self.p.trg_dim])

        # first layer (from input into hidden)
        hidden_output = self._add_layer(self._src_embeddings, self.p.src_dim, self.p.map_dim, 1)
        # further hidden layers (staying in hidden)
        for i in range(self.p.num_layers - 1):
            hidden_output = self._add_layer(hidden_output, self.p.map_dim, self.p.map_dim, i+2)

        # linear map out of the hidden space
        map_trg_matr = tf.Variable(
            tf.random_normal([self.p.map_dim, self.p.trg_dim], stddev=1),
            name="map_trg_matr"
        )
        map_trg_bias = tf.Variable(
            tf.zeros([self.p.trg_dim]),
            name="map_trg_bias"
        )

        # output embeddings
        self._mapped_target = tf.matmul(
            hidden_output,
            map_trg_matr
        ) + map_trg_bias

        # MSE training
        loss = tf.nn.l2_loss(
            self._mapped_target - self._trg_embeddings
        )
        self._mse = tf.reduce_mean(loss, name="MSE")

        optimizer = tf.train.AdamOptimizer()
        self._train_step = optimizer.minimize(loss)

        # initialization
        self._init = tf.global_variables_initializer()

    def _add_layer(self, inputs, in_dim, out_dim, layer_num):
        with tf.variable_scope("layer_%d" % layer_num):
            weights = tf.Variable(
                tf.random_normal([in_dim, out_dim], stddev=1),
                name="weights"
            )
            bias = tf.Variable(
                tf.zeros([out_dim]),
                name="biases"
            )

            layer_input = tf.matmul(
                inputs,
                weights
            ) + bias

            if self.p.activation == 'tanh':
                activation = tf.nn.tanh(layer_input, name="tanh_activation")
            elif self.p.activation == 'relu':
                activation = tf.nn.relu(layer_input, name="relu_activation")

        return activation

    def train_batch(self, batch_src, batch_trg):
        (loss, _) = self._session.run(
            [self._mse, self._train_step],
            feed_dict = {
                self._src_embeddings : batch_src,
                self._trg_embeddings : batch_trg
            }
        )
        return loss

    def eval_batch(self, batch_src, batch_trg):
        (loss,) = self._session.run(
            [self._mse],
            feed_dict = {
                self._src_embeddings : batch_src,
                self._trg_embeddings : batch_trg
            }
        )
        return loss

    def project_batch(self, batch_src):
        (batch_proj,) = self._session.run(
            [self._mapped_target],
            feed_dict = {
                self._src_embeddings : batch_src
            }
        )
        return batch_proj
