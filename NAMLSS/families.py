import tensorflow_probability as tfp
import tensorflow as tf


class Gaussian:

    def __init__(self, two_param=True):
        self._two_param = two_param

    def loss(self, loc, scale, val):

        if self._two_param:
            dist = tfp.distributions.Normal(loc=loc, scale=tf.sqrt(tf.exp(scale)))
            out = -tf.reduce_sum(dist.log_prob(value=val))
        else:
            mse = tf.keras.losses.MeanSquaredError()
            out = mse(val, loc)
        return out

    def log_likelihood(self, loc, scale, val):

        if self._two_param:
            dist = tfp.distributions.Normal(loc=loc, scale=tf.exp(scale))
            out = tf.reduce_sum(dist.log_prob(value=val))

        return out
