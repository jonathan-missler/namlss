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
            dist = tfp.distributions.Normal(loc=loc, scale=tf.sqrt(tf.exp(scale)))
            out = tf.reduce_sum(dist.log_prob(value=val))
        else:
            dist = tfp.distributions.Normal(loc=loc, scale=tfp.stats.stddev(val))
            out = tf.reduce_sum(dist.log_prob(value=val))

        return out


class Gamma:

    def __init__(self, log_rate=False):
        self._log_rate = log_rate
        self.two_param = True

    def loss(self, conc, rate, val):

        if self._log_rate:
            dist = tfp.distributions.Gamma(concentration=tf.exp(conc), log_rate=rate)
        else:
            dist = tfp.distributions.Gamma(concentration=tf.exp(conc), rate=tf.exp(rate))

        out = -tf.reduce_sum(dist.log_prob(value=val))
        return out

    def log_likelihood(self, conc, rate, val):
        return -self.loss(conc, rate, val)


class InvGauss:

    def __init__(self):
        self.two_param = True

    def loss(self, loc, shape, val):
        dist = tfp.distributions.InverseGaussian(loc=(1/tf.sqrt(tf.math.softplus(loc))), concentration=tf.exp(shape))
        out = -tf.reduce_sum(dist.log_prob(value=val))
        return out

    def log_likelihood(self, loc, shape, val):
        return -self.loss(loc, shape, val)
