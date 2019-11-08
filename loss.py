import tensorflow as tf


def categorical_loss(y_true, y_pred, lambd):
    loss = tf.losses.softmax_cross_entropy(y_true, y_pred) * lambd
    return loss


def gan_loss(y_true, y_pred):
    loss = tf.losses.mean_squared_error(y_true, y_pred)
    return loss


def cycle_loss(y_true, y_pred, lambd):
    loss = tf.losses.absolute_difference(y_true, y_pred) * lambd
    return loss
