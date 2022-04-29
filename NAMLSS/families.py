import tensorflow_probability as tfp
import tensorflow as tf


class Gaussian:

    def __init__(self, estimate_scale=True):
        self.est_scale = estimate_scale

    def loss(self, loc, scale, val):

        if self.est_scale:
            dist = tfp.distributions.Normal(loc=loc, scale=tf.exp(scale))
            out = -tf.reduce_sum(dist.log_prob(value=val))
        else:
            mse = tf.keras.losses.MeanSquaredError()
            out = mse(val, loc)
        return out
