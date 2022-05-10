import tensorflow as tf


class Trainer:

    def __init__(self, model, family, config):
        self.model = model
        self.family = family
        self.config = config

    def loss(self, x, y, training):
        loc, scale = self.model(x, training=training)
        out = self.family.loss(loc, scale, y)

        return out

    def grad(self, x, y):
        with tf.GradientTape() as tape:
            loss_val = self.loss(x, y, training=True)
        return loss_val, tape.gradient(loss_val, self.model.trainable_variables)
