import numpy as np
from tqdm import trange
import os
import sys
import tensorflow as tf

from tensorflow.contrib import rnn


utils_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils_dir = os.path.join(utils_dir, 'utils')
sys.path.append(utils_dir)

from model_utils import batch_data, batch_data_multiple_iters
from tf_utils import graph_size, process_grad



def process_x(raw_x):
    """
    raw_x: list of numpy arrays with shape [seq_len, num_features]
    """
    return np.array(raw_x, dtype=np.float32)

def process_y(raw_y):
    """
    raw_y: list of numpy arrays with shape [pred_len, num_targets]
    """
    return np.array(raw_y, dtype=np.float32)



def temporal_attention(inputs, hidden_size):
    """
    inputs: [B, T, H]
    returns: [B, H]
    """
    with tf.variable_scope("temporal_attention"):
        w = tf.get_variable("w", [hidden_size, hidden_size])
        b = tf.get_variable("b", [hidden_size])
        u = tf.get_variable("u", [hidden_size])

        v = tf.tanh(tf.tensordot(inputs, w, axes=1) + b)
        vu = tf.tensordot(v, u, axes=1)
        alphas = tf.nn.softmax(vu)

        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), axis=1)
        return output



class Model(object):
    def __init__(self,
                 seq_len=168,
                 pred_len=24,
                 num_features=7,
                 num_targets=5,
                 n_hidden=128,
                 optimizer=None,
                 seed=1):

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.num_targets = num_targets
        self.n_hidden = n_hidden

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = \
                self.create_model(optimizer)
            self.saver = tf.train.Saver()

        self.sess = tf.Session(graph=self.graph)
        self.size = graph_size(self.graph)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(
                self.graph, run_meta=metadata, cmd='scope', options=opts
            ).total_float_ops

    def create_model(self, optimizer):

        features = tf.placeholder(
            tf.float32, [None, self.seq_len, self.num_features], name="features"
        )
        labels = tf.placeholder(
            tf.float32, [None, self.pred_len, self.num_targets], name="labels"
        )

        # -------- LSTM Encoder --------
        stacked_lstm = rnn.MultiRNNCell(
            [rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)]
        )

        outputs, _ = tf.nn.dynamic_rnn(
            stacked_lstm, features, dtype=tf.float32
        )

        # -------- Temporal Attention --------
        context = temporal_attention(outputs, self.n_hidden)

        # -------- Prediction Head --------
        pred = tf.layers.dense(
            context, self.pred_len * self.num_targets
        )
        pred = tf.reshape(
            pred, [-1, self.pred_len, self.num_targets]
        )

        # -------- Loss --------
        loss = tf.losses.mean_squared_error(labels, pred)

        # -------- Optimization --------
        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars)

        # -------- Evaluation --------
        eval_metric_ops = tf.reduce_mean(tf.abs(pred - labels))

        return features, labels, train_op, grads, eval_metric_ops, loss



    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):
        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        processed_samples = 0

        if num_samples < 32:
            x = process_x(data['x'])
            y = process_y(data['y'])
            with self.graph.as_default():
                model_grads = self.sess.run(
                    self.grads,
                    feed_dict={self.features: x, self.labels: y}
                )
            grads = process_grad(model_grads)
            processed_samples = num_samples

        else:
            for i in range(min(num_samples // 32, 4)):
                x = process_x(data['x'][32*i:32*(i+1)])
                y = process_y(data['y'][32*i:32*(i+1)])
                with self.graph.as_default():
                    model_grads = self.sess.run(
                        self.grads,
                        feed_dict={self.features: x, self.labels: y}
                    )
                grads += process_grad(model_grads)

            grads /= min(num_samples // 32, 4)
            processed_samples = min(num_samples // 32, 4) * 32

        return processed_samples, grads

    def solve_inner(self, data, num_epochs=1, batch_size=32):

        with self.graph.as_default():
            _, grads = self.get_gradients(data, self.size)

        for _ in trange(num_epochs, desc='Epoch', leave=False):
            for X, y in batch_data(data, batch_size):
                x = process_x(X)
                y = process_y(y)
                with self.graph.as_default():
                    self.sess.run(
                        selshradha
                        f.train_op,
                        feed_dict={self.features: x, self.labels: y}
                    )

        soln = self.get_params()
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size * self.flops
        return soln, comp, grads

    def solve_iters(self, data, num_iters=1, batch_size=32):

        for X, y in batch_data_multiple_iters(data, batch_size, num_iters):
            x = process_x(X)
            y = process_y(y)
            with self.graph.as_default():
                self.sess.run(
                    self.train_op,
                    feed_dict={self.features: x, self.labels: y}
                )

        soln = self.get_params()
        return soln, 0

    def test(self, data):

        x = process_x(data['x'])
        y = process_y(data['y'])

        with self.graph.as_default():
            metric, loss = self.sess.run(
                [self.eval_metric_ops, self.loss],
                feed_dict={self.features: x, self.labels: y}
            )

        return metric, loss

    def close(self):
        self.sess.close()
