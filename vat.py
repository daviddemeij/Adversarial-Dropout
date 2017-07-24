import tensorflow as tf
import numpy as np
import sys, os

import layers as L
import cnn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('epsilon', 16.0, "norm length for (virtual) adversarial training ")
tf.app.flags.DEFINE_integer('num_power_iterations', 1, "the number of power iterations")
tf.app.flags.DEFINE_float('xi', 1e-6, "small constant for finite difference")
tf.app.flags.DEFINE_float('dropout_delta', 0.2, "small constant for finite difference")



def logit(x, is_training=True, update_batch_stats=True, stochastic=True, seed=1234, dropout_mask=None, return_mask=False):
    return cnn.logit(x, is_training=is_training,
                     update_batch_stats=update_batch_stats,
                     stochastic=stochastic,
                     seed=seed, dropout_mask=dropout_mask, return_mask=return_mask)


def forward(x, dropout_mask=None, is_training=True, update_batch_stats=True, seed=1234, return_mask=False):
    if is_training:
        return logit(x, is_training=True,
                     update_batch_stats=update_batch_stats,
                     stochastic=True, seed=seed, dropout_mask=dropout_mask, return_mask=return_mask)
    else:
        return logit(x, is_training=False,
                     update_batch_stats=update_batch_stats,
                     stochastic=False, seed=seed, return_mask=return_mask)


def get_normalized_vector(d):
    d /= (1e-12 + tf.reduce_max(tf.abs(d), range(1, len(d.get_shape())), keep_dims=True))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), range(1, len(d.get_shape())), keep_dims=True))
    return d


def generate_virtual_adversarial_perturbation(x, logit, is_training=True):
    d = tf.random_normal(shape=tf.shape(x))
    for _ in range(FLAGS.num_power_iterations):
        d = FLAGS.xi * get_normalized_vector(d)
        logit_p = logit
        fts_m, logit_m = forward(x + d, update_batch_stats=False, is_training=is_training)
        dist = L.kl_divergence_with_logit(logit_p, logit_m)
        grad = tf.gradients(dist, [d], aggregation_method=2)[0]
        d = tf.stop_gradient(grad)

    return FLAGS.epsilon * get_normalized_vector(d)

def generate_adversarial_dropout_mask(x, logit, original_dropout_mask, update_fraction=0.2):
    """ Calculate the dropout mask for which the logit changes the most while not changing more then
    a given fraction (update_fraction) of the mask. """
    dropout_ones = tf.ones(tf.shape(original_dropout_mask))
    logit_without_dropout = forward(x, dropout_mask=dropout_ones)
    dist = tf.reduce_mean(tf.squared_difference(logit, logit_without_dropout), axis=1)
    jacobian = tf.gradients(dist, [dropout_ones])[0]
    number_of_updates = tf.cast(update_fraction * 16384 * 2, tf.int32)
    with tf.device('/cpu:0'):
        highest_jacobian_values = tf.nn.top_k(tf.abs(jacobian), number_of_updates)
    with tf.device('/gpu:0'):
        minimum_value = tf.reduce_min(highest_jacobian_values.values, axis=1)
        top_jacobian = tf.transpose(tf.greater_equal(tf.transpose(tf.abs(jacobian)), tf.transpose(minimum_value)))
        jacobian_below_zero = tf.greater_equal(jacobian, 0.0)
        zero_or_one = tf.where(jacobian_below_zero, tf.ones_like(jacobian), tf.zeros_like(jacobian))
        dropout_mask_copy = tf.where(top_jacobian, zero_or_one, original_dropout_mask)
    #fraction_change = (1.0 / 16384.0) * (tf.reduce_sum(tf.square(dropout_mask - dropout_mask_copy), axis=[1]))
    #dropout_mask_percentage = tf.reduce_sum(dropout_mask_copy, axis=1) / 16384.0
    return dropout_mask_copy

def vad_loss(x, logit, dropout_mask, is_training=True, name="vad_loss", update_fraction=0.2):
    """ Generate Virtual Adversarial Dropout loss by updating the dropout mask towards the adversarial direction
    and comparing the logit using this dropout mask with the regular logit """
    adv_mask = generate_adversarial_dropout_mask(x, logit, dropout_mask)
    logit_adv = forward(x, update_batch_stats=False, is_training=is_training, dropout_mask=adv_mask)
    loss = tf.reduce_mean(tf.squared_difference(logit, logit_adv), axis=[0,1])
    return loss

def virtual_adversarial_loss(x, logit, is_training=True, name="vat_loss"):
    r_vadv = generate_virtual_adversarial_perturbation(x, logit, is_training=is_training)
    logit = tf.stop_gradient(logit)
    logit_p = logit
    logit_m = forward(x + r_vadv, update_batch_stats=False, is_training=is_training)
    loss = L.kl_divergence_with_logit(logit_p, logit_m)
    return tf.identity(loss, name=name)


# Regular Adversarial Training
def generate_adversarial_perturbation(x, loss):
    grad = tf.gradients(loss, [x], aggregation_method=2)[0]
    grad = tf.stop_gradient(grad)
    return FLAGS.epsilon * get_normalized_vector(grad)


def adversarial_loss(x, y, loss, is_training=True, name="at_loss"):
    r_adv = generate_adversarial_perturbation(x, loss)
    logit = forward(x + r_adv, is_training=is_training, update_batch_stats=False)
    loss = L.ce_loss(logit, y)
    return loss
