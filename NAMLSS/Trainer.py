import tensorflow as tf


class Trainer:

    def __init__(self, model, family, optimizer, config):
        self.model = model
        self.family = family
        self.config = config
        self.optimizer = optimizer

    def loss(self, x, y, training):
        loc, scale = self.model(x, training=training)
        out = self.family.loss(loc, scale, y)

        return out

    def grad(self, x, y):
        with tf.GradientTape() as tape:
            loss_val = self.loss(x, y, training=True)
        return loss_val, tape.gradient(loss_val, self.model.trainable_variables)

    def train_epoch(self, train_batch):
        for x, y in train_batch:
            loss_val, grads = self.grad(x, y)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            self.epoch_loss_avg.update_state(loss_val)

    def run_training(self, train_batch, val_batch):
        train_loss_results = []

        num_epochs = self.config.num_epochs

        for epoch in range(num_epochs):
            self.epoch_loss_avg = tf.keras.metrics.Mean()

            self.train_epoch(train_batch)

            train_loss_results.append(self.epoch_loss_avg.result())

            if epoch % 5 == 0:
                print("Epoch {:03d}: Loss: {:.3f}".format(epoch,
                                                          self.epoch_loss_avg.result()))

        return train_loss_results
