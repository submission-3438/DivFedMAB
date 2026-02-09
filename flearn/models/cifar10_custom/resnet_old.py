 


import numpy as np
import tensorflow as tf
from flearn.utils.model_utils import batch_data
from flearn.utils.tf_utils import graph_size, process_grad


# ======================================================
# Gradient projection configuration
# ======================================================
# ResNet-18 full gradient ≈ 11M params
# STRIDE = 21 → ≈ 523K projected dimensions
GRAD_PROJ_STRIDE = 21


def conv3x3(x, filters, stride):
    return tf.layers.conv2d(
        x, filters, kernel_size=3, strides=stride,
        padding='same', use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer()
    )


def batch_norm_relu(x, training):
    x = tf.layers.batch_normalization(x, training=training)
    return tf.nn.relu(x)


def residual_block(x, filters, stride, training):
    shortcut = x

    x = conv3x3(x, filters, stride)
    x = batch_norm_relu(x, training)

    x = conv3x3(x, filters, 1)
    x = tf.layers.batch_normalization(x, training=training)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.layers.conv2d(
            shortcut, filters, kernel_size=1,
            strides=stride, padding='same',
            use_bias=False
        )
        shortcut = tf.layers.batch_normalization(shortcut, training=training)

    return tf.nn.relu(x + shortcut)


class Model(object):
    def __init__(self, num_classes, optimizer, seed=1):
        self.num_classes = num_classes

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + seed)
            self.features, self.labels, self.train_op, self.grads, \
            self.eval_metric_ops, self.loss, self.predictions = \
                self.create_model(optimizer)

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
    # Model definition
    # ======================================================
    def create_model(self, optimizer):
        features = tf.placeholder(tf.float32, [None, 32, 32, 3], name='features')
        labels = tf.placeholder(tf.int64, [None], name='labels')
        training = tf.placeholder_with_default(False, shape=())

        x = conv3x3(features, 64, 1)
        x = batch_norm_relu(x, training)

        # ResNet-18 blocks
        for _ in range(2):
            x = residual_block(x, 64, 1, training)

        for _ in range(2):
            x = residual_block(x, 128, 2 if _ == 0 else 1, training)

        for _ in range(2):
            x = residual_block(x, 256, 2 if _ == 0 else 1, training)

        for _ in range(2):
            x = residual_block(x, 512, 2 if _ == 0 else 1, training)

        x = tf.reduce_mean(x, axis=[1, 2])  # Global Average Pooling
        logits = tf.layers.dense(x, self.num_classes)

        predictions = tf.argmax(logits, axis=1)
        loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars)

        correct = tf.equal(labels, predictions)
        eval_metric_ops = tf.reduce_sum(tf.cast(correct, tf.int32))

        return features, labels, train_op, grads, eval_metric_ops, loss, predictions

    # ======================================================
    # Federated Learning hooks
    # ======================================================
    def set_params(self, model_params):
        with self.graph.as_default():
            for var, val in zip(tf.trainable_variables(), model_params):
                var.load(val, self.sess)

    def get_params(self):
        with self.graph.as_default():
            return self.sess.run(tf.trainable_variables())

    def get_gradients(self, mini_batch_data, model_len=None):
        """
        Returns projected gradients for submodular selection.
        Projection is deterministic and shared across all clients.
        """
        with self.graph.as_default():
            grads = self.sess.run(
                self.grads,
                feed_dict={
                    self.features: mini_batch_data['x'],
                    self.labels: mini_batch_data['y']
                }
            )

        flat = process_grad(grads)

        # ---- PROJECTION STEP (CRITICAL FIX) ----
        proj = flat[::GRAD_PROJ_STRIDE]

        return len(mini_batch_data['y']), proj

    def solve_inner(self, data, num_epochs=1, batch_size=32):
        for _ in range(num_epochs):
            for X, y in batch_data(data, batch_size):
                self.sess.run(
                    self.train_op,
                    feed_dict={self.features: X, self.labels: y}
                )

        soln = self.get_params()
        grads = self.get_gradients(data)[1]
        comp = num_epochs * self.flops
        return soln, comp, grads

    def test(self, data):
        return self.sess.run(
            [self.eval_metric_ops, self.loss],
            feed_dict={self.features: data['x'], self.labels: data['y']}
        )

    def get_loss(self, data):
        return self.sess.run(
            self.loss,
            feed_dict={self.features: data['x'], self.labels: data['y']}
        )

    def close(self):
        self.sess.close()
































