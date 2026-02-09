# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from flearn.utils.model_utils import batch_data
from flearn.utils.tf_utils import graph_size, process_grad

# ======================================================
# Gradient projection configuration
# ======================================================
GRAD_PROJ_STRIDE = 21


# ======================================================
# Layers
# ======================================================
def conv3x3(x, filters, stride):
    return tf.layers.conv2d(
        x,
        filters,
        kernel_size=3,
        strides=stride,
        padding='same',
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer()
    )


def group_norm_relu(x, groups=8, eps=1e-5):
    """
    Manual Group Normalization for TF-1.x
    """
    with tf.variable_scope(None, default_name="group_norm"):
        N, H, W, C = x.get_shape().as_list()
        G = min(groups, C)

        x = tf.reshape(x, [-1, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        x = tf.reshape(x, [-1, H, W, C])

        gamma = tf.get_variable("gamma", [C], initializer=tf.ones_initializer())
        beta  = tf.get_variable("beta",  [C], initializer=tf.zeros_initializer())

        x = gamma * x + beta
        return tf.nn.relu(x)


def group_norm(x, groups=8, eps=1e-5):
    """
    GroupNorm without ReLU (used before residual add)
    """
    with tf.variable_scope(None, default_name="group_norm"):
        N, H, W, C = x.get_shape().as_list()
        G = min(groups, C)

        x = tf.reshape(x, [-1, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        x = tf.reshape(x, [-1, H, W, C])

        gamma = tf.get_variable("gamma", [C], initializer=tf.ones_initializer())
        beta  = tf.get_variable("beta",  [C], initializer=tf.zeros_initializer())

        return gamma * x + beta


def residual_block(x, filters, stride):
    shortcut = x

    x = conv3x3(x, filters, stride)
    x = group_norm_relu(x)

    x = conv3x3(x, filters, 1)
    x = group_norm(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.layers.conv2d(
            shortcut,
            filters,
            kernel_size=1,
            strides=stride,
            padding='same',
            use_bias=False
        )
        shortcut = group_norm(shortcut)

    return tf.nn.relu(x + shortcut)


# ======================================================
# Model
# ======================================================
class Model(object):
    def __init__(self, num_classes, optimizer, seed=1):
        self.num_classes = num_classes

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + seed)

            (self.features,
             self.labels,
             self.training,
             self.train_op,
             self.grads,
             self.eval_metric_ops,
             self.loss,
             self.predictions) = self.create_model(optimizer)

            self.saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        # model size and flops
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(
                self.graph, cmd='scope', options=opts
            ).total_float_ops

    # ======================================================
    # Architecture
    # ======================================================
    def create_model(self, optimizer):
        features = tf.placeholder(tf.float32, [None, 32, 32, 3], name='features')
        labels = tf.placeholder(tf.int64, [None], name='labels')
        training = tf.placeholder(tf.bool, name='training')

        # --------------------------------------------------
        # CIFAR-100 normalization
        # --------------------------------------------------
        mean = tf.constant([0.5071, 0.4867, 0.4408])
        std  = tf.constant([0.2675, 0.2565, 0.2761])
        x = (features / 255.0 - mean) / std

        # --------------------------------------------------
        # Stem
        # --------------------------------------------------
        x = conv3x3(x, 16, 1)
        x = group_norm_relu(x)

        # --------------------------------------------------
        # ResNet-32 (CIFAR style)
        # --------------------------------------------------
        for _ in range(5):
            x = residual_block(x, 16, 1)

        for i in range(5):
            x = residual_block(x, 32, 2 if i == 0 else 1)

        for i in range(5):
            x = residual_block(x, 64, 2 if i == 0 else 1)

        # --------------------------------------------------
        # Head
        # --------------------------------------------------
        x = tf.reduce_mean(x, axis=[1, 2])
        logits = tf.layers.dense(
            x,
            self.num_classes,
            kernel_regularizer=tf.keras.regularizers.l2(5e-4)
        )

        predictions = tf.argmax(logits, axis=1)
        loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)

        train_op = optimizer.apply_gradients(grads_and_vars)

        correct = tf.equal(labels, predictions)
        eval_metric_ops = tf.reduce_sum(tf.cast(correct, tf.int32))

        return (features,
                labels,
                training,
                train_op,
                grads,
                eval_metric_ops,
                loss,
                predictions)

    # ======================================================
    # Federated Learning
    # ======================================================
    def set_params(self, model_params):
        with self.graph.as_default():
            for var, val in zip(tf.trainable_variables(), model_params):
                var.load(val, self.sess)

    def get_params(self):
        with self.graph.as_default():
            return self.sess.run(tf.trainable_variables())

    def get_gradients(self, mini_batch_data, model_len=None):
        with self.graph.as_default():
            grads = self.sess.run(
                self.grads,
                feed_dict={
                    self.features: mini_batch_data['x'],
                    self.labels: mini_batch_data['y'],
                    self.training: False
                }
            )

        flat = process_grad(grads)
        proj = flat[::GRAD_PROJ_STRIDE]

        return len(mini_batch_data['y']), proj

    def solve_inner(self, data, num_epochs=1, batch_size=32):
        for _ in range(num_epochs):
            for X, y in batch_data(data, batch_size):
                self.sess.run(
                    self.train_op,
                    feed_dict={
                        self.features: X,
                        self.labels: y,
                        self.training: True
                    }
                )

        soln = self.get_params()
        grads = self.get_gradients(data)[1]
        comp = num_epochs * self.flops
        return soln, comp, grads

    def test(self, data):
        return self.sess.run(
            [self.eval_metric_ops, self.loss],
            feed_dict={
                self.features: data['x'],
                self.labels: data['y'],
                self.training: False
            }
        )

    def get_loss(self, data):
        return self.sess.run(
            self.loss,
            feed_dict={
                self.features: data['x'],
                self.labels: data['y'],
                self.training: False
            }
        )

    def close(self):
        self.sess.close()
