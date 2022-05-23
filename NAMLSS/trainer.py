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

            self.epoch_train_loss_avg.update_state(loss_val)

    def val_epoch(self, val_batch):
        for x, y in val_batch:
            loss_val = self.loss(x, y, training=False)

            self.epoch_val_loss_avg.update_state(loss_val)

    def run_training(self, train_batch, val_batch):
        train_loss_results = []
        val_loss_results = []

        num_epochs = self.config.num_epochs

        for epoch in range(num_epochs):
            self.epoch_train_loss_avg = tf.keras.metrics.Mean()
            self.epoch_val_loss_avg = tf.keras.metrics.Mean()

            self.train_epoch(train_batch)
            self.val_epoch(val_batch)

            train_loss_results.append(self.epoch_train_loss_avg.result())
            val_loss_results.append(self.epoch_val_loss_avg.result())

            print("Epoch {:03d}: Train Loss: {:.3f} Validation Loss: {:.3f}".format(epoch+1,
                                                                                    self.epoch_train_loss_avg.result(),
                                                                                    self.epoch_val_loss_avg.result()))

        return train_loss_results, val_loss_results
