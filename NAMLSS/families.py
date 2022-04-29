import tensorflow_probability as tfp


class Gaussian():

    def __init__(self, estimate_scale = True):
        self.est_scale = estimate_scale

    def loss(self, loc, scale, val):


