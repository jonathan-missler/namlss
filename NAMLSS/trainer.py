import os

import tensorflow as tf
import numpy as np
from neural_additive_models.graph_builder import weight_decay, feature_output_regularization
from nam.utils.loggers import TensorBoardLogger


class Trainer:

    def __init__(self, model, family, optimizer, config, logger=True, checkpoints=True):
        self.model = model
        self.family = family
        self.config = config
        self.optimizer = optimizer

        if logger:
            self.logger = TensorBoardLogger(self.config)

        if checkpoints:
            self.checkpointer = Checkpointer(self.model, config)

    def loss(self, x, y, training):
        loc, scale = self.model(x, training=training)
        out = self.family.loss(loc, scale, y)
        return out

    def penalized_loss(self, x, y, training):
        loss = self.loss(x, y, training=training)
        reg_loss = 0.0

        if self.config.output_regularization1 > 0:
            reg_loss += self.config.output_regularization1 * feature_output_regularization(self.model.mod1, x)

        if self.config.output_regularization2 > 0:
            reg_loss += self.config.output_regularization2 * feature_output_regularization(self.model.mod2, x)

        if self.config.l2_regularization1 > 0:
            num_networks1 = len(self.model.mod1.feature_nns)
            reg_loss += self.config.l2_regularization1 * weight_decay(self.model.mod1, num_networks=num_networks1)

        if self.config.l2_regularization2 > 0:
            num_networks2 = len(self.model.mod2.feature_nns)
            reg_loss += self.config.l2_regularization2 * weight_decay(self.model.mod2, num_networks=num_networks2)

        return loss + reg_loss

    def grad(self, x, y):
        with tf.GradientTape() as tape:
            loss_val = self.penalized_loss(x, y, training=True)
        return loss_val, tape.gradient(loss_val, self.model.trainable_variables)

    def train_epoch(self, train_batch):
        epoch_train_loss = []
        for x, y in train_batch:
            loss_val, grads = self.grad(x, y)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            epoch_train_loss.append(loss_val/len(y))

        return np.mean(epoch_train_loss)

    def val_epoch(self, val_batch):
        epoch_val_loss = []
        for x, y in val_batch:
            loss_val = self.penalized_loss(x, y, training=True)

            epoch_val_loss.append(loss_val/len(y))

        return np.mean(epoch_val_loss)

    def run_training(self, train_batch, val_batch):
        train_loss_results = []
        val_loss_results = []

        num_epochs = self.config.num_epochs

        for epoch in range(num_epochs):

            epoch_train_loss_avg = self.train_epoch(train_batch)
            epoch_val_loss_avg = self.val_epoch(val_batch)

            if self.logger:
                self.logger.write({"train_loss_epoch": epoch_train_loss_avg,
                                   "val_loss_epoch": epoch_val_loss_avg})

            if epoch % self.config.save_frequency == 0:
                if self.checkpointer:
                    self.checkpointer.save(epoch)

            train_loss_results.append(epoch_train_loss_avg)
            val_loss_results.append(epoch_val_loss_avg)

            print("Epoch {:03d}: Train Loss: {:.3f} Validation Loss: {:.3f}".format(epoch+1,
                                                                                    epoch_train_loss_avg,
                                                                                    epoch_val_loss_avg))

        return train_loss_results, val_loss_results


class Checkpointer:

    def __init__(self, model, config):
        self.model = model
        self.config = config

        self._checkpt_dir = os.path.join(self.config.logdir, "checkpts")
        os.makedirs(self._checkpt_dir, exist_ok=True)

    def save(self, epoch):
        checkpt_path = os.path.join(self._checkpt_dir, "{}-{}.pt".format(self.config.name_scope, epoch))

        self.model.save_weights(checkpt_path)

    def load(self, epoch):
        checkpt_path = os.path.join(self._checkpt_dir, "{}-{}.pt".format(self.config.name_scope, epoch))

        self.model.load_weights(checkpt_path)
