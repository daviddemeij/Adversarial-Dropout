import tensorflow as tf
import numpy as np
import layers as L

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('keep_prob_hidden', 0.5, "dropout rate")
tf.app.flags.DEFINE_float('lrelu_a', 0.1, "lrelu slope")
tf.app.flags.DEFINE_boolean('top_bn', False, "")
X_DIM = 2
NUM_CLASSES = 2
def logit_moons(x, is_training=True, update_batch_stats=True, stochastic=True, seed=1234, dropout_mask=None):
    h = x
    rng = np.random.RandomState(seed)
    h = L.fc(h, dim_in=X_DIM, dim_out=64, seed=rng.randint(123456), name='fc1')
    h = L.lrelu(h, FLAGS.lrelu_a)
    h = L.fc(h, dim_in=64, dim_out=64, seed=rng.randint(123456), name='fc2')
    h = L.lrelu(h, FLAGS.lrelu_a)
    if stochastic:
        if dropout_mask == None:
            dropout_mask = tf.cast(
                tf.greater_equal(tf.random_uniform(tf.shape(h), 0, 1, seed=rng.randint(123456)), 1.0 - FLAGS.keep_prob_hidden),
                tf.float32)
        else:
            dropout_mask = tf.reshape(dropout_mask, tf.shape(h))
        h = tf.multiply(h, dropout_mask)
        h = (1.0 / FLAGS.keep_prob_hidden) * h

    h = L.fc(h, dim_in=64, dim_out=NUM_CLASSES, seed=rng.randint(123456), name='fc3')
    return h


def logit(x, is_training=True, update_batch_stats=True, stochastic=True, seed=1234, dropout_mask=None, return_mask=False):
    h = x

    rng = np.random.RandomState(seed)

    h = L.conv(h, ksize=3, stride=1, f_in=3, f_out=128, seed=rng.randint(123456), name='c1')
    h = L.lrelu(L.bn(h, 128, is_training=is_training, update_batch_stats=update_batch_stats, name='b1'), FLAGS.lrelu_a)
    h = L.conv(h, ksize=3, stride=1, f_in=128, f_out=128, seed=rng.randint(123456), name='c2')
    h = L.lrelu(L.bn(h, 128, is_training=is_training, update_batch_stats=update_batch_stats, name='b2'), FLAGS.lrelu_a)
    h = L.conv(h, ksize=3, stride=1, f_in=128, f_out=128, seed=rng.randint(123456), name='c3')
    h = L.lrelu(L.bn(h, 128, is_training=is_training, update_batch_stats=update_batch_stats, name='b3'), FLAGS.lrelu_a)

    h = L.max_pool(h, ksize=2, stride=2)
    if stochastic:
        h = tf.nn.dropout(h, keep_prob=FLAGS.keep_prob_hidden)

    h = L.conv(h, ksize=3, stride=1, f_in=128, f_out=256, seed=rng.randint(123456), name='c4')
    h = L.lrelu(L.bn(h, 256, is_training=is_training, update_batch_stats=update_batch_stats, name='b4'), FLAGS.lrelu_a)
    h = L.conv(h, ksize=3, stride=1, f_in=256, f_out=256, seed=rng.randint(123456), name='c5')
    h = L.lrelu(L.bn(h, 256, is_training=is_training, update_batch_stats=update_batch_stats, name='b5'), FLAGS.lrelu_a)
    h = L.conv(h, ksize=3, stride=1, f_in=256, f_out=256, seed=rng.randint(123456), name='c6')
    h = L.lrelu(L.bn(h, 256, is_training=is_training, update_batch_stats=update_batch_stats, name='b6'), FLAGS.lrelu_a)

    h = L.max_pool(h, ksize=2, stride=2)

    # Making it possible to change or return a dropout mask
    if stochastic:
        if dropout_mask == None:
            dropout_mask = tf.cast(
                tf.greater_equal(tf.random_uniform(tf.shape(h), 0, 1, seed=rng.randint(123456)), 1.0 - FLAGS.keep_prob_hidden),
                tf.float32)
        else:
            dropout_mask = tf.reshape(dropout_mask, tf.shape(h))
        h = tf.multiply(h, dropout_mask)
        h = (1.0 / FLAGS.keep_prob_hidden) * h

    h = L.conv(h, ksize=3, stride=1, f_in=256, f_out=512, seed=rng.randint(123456), padding="VALID", name='c7')
    h = L.lrelu(L.bn(h, 512, is_training=is_training, update_batch_stats=update_batch_stats, name='b7'), FLAGS.lrelu_a)
    h = L.conv(h, ksize=1, stride=1, f_in=512, f_out=256, seed=rng.randint(123456), name='c8')
    h = L.lrelu(L.bn(h, 256, is_training=is_training, update_batch_stats=update_batch_stats, name='b8'), FLAGS.lrelu_a)
    h = L.conv(h, ksize=1, stride=1, f_in=256, f_out=128, seed=rng.randint(123456), name='c9')
    h = L.lrelu(L.bn(h, 128, is_training=is_training, update_batch_stats=update_batch_stats, name='b9'), FLAGS.lrelu_a)

    h = tf.reduce_mean(h, reduction_indices=[1, 2])  # Global average pooling
    h = L.fc(h, 128, 10, seed=rng.randint(123456), name='fc')

    if FLAGS.top_bn:
        h = L.bn(h, 10, is_training=is_training,
                 update_batch_stats=update_batch_stats, name='bfc')
    if return_mask:
        return h, tf.reshape(dropout_mask, [-1, 8*8*256])
    else:
        return h
