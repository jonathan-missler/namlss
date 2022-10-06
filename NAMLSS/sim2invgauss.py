import tensorflow as tf
import pandas as pd
from NAMLSS.model import NamLSS
from NAMLSS.trainer import Trainer
from NAMLSS.config import defaults
from NAMLSS.families import InvGauss
from NAMLSS.simdata import sim_invgauss
import numpy as np
from neural_additive_models.data_utils import split_training_dataset
import matplotlib.pyplot as plt

config = defaults()
config.batch_size = 300

y, x1, x2, x3 = sim_invgauss()

data_array = np.hstack((y.numpy(), x1.numpy(), x2.numpy(), x3.numpy()))

split_generator = split_training_dataset(data_array[:, 1:], data_array[:, 0], n_splits=1, stratified=False)

for i in split_generator:
    (train_features, train_target), (val_features, val_target) = i


train_data = tf.data.Dataset.from_tensor_slices((train_features, train_target))
val_data = tf.data.Dataset.from_tensor_slices((val_features, val_target))

train_batches = train_data.shuffle(1000).batch(config.batch_size)
val_batches = val_data.shuffle(1000).batch(config.batch_size)

num_unique_vals = [
    len(np.unique(train_features[:, i])) for i in range(train_features.shape[1])
]
num_units = [
    min(config.num_basis_functions, i * config.units_multiplier) for i in num_unique_vals
]
num_inputs = train_features.shape[-1]

config.activation = "relu"
config.shallow = False
config.num_epochs = 500
config.lr = 0.001
config.dropout = 0.1
config.feature_dropout = 0.1

config.output_regularization1 = 0.01
config.output_regularization2 = 0.01
config.l2_regularization1 = 0.01
config.l2_regularization2 = 0.01
config.early_stopping_patience = 15

family = InvGauss()
model = NamLSS(num_inputs=num_inputs, num_units=num_units, family=family, feature_dropout=config.feature_dropout,
               dropout=config.dropout, shallow=config.shallow, activation=config.activation)
optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)
trainer = Trainer(model, family, optimizer, config)

train_losses, val_losses = trainer.run_training(train_batches, val_batches)

loc_pred = trainer.model.mod1.calc_outputs(train_features, training=False)
loc_pred = tf.exp(loc_pred)

shape_pred = trainer.model.mod2.calc_outputs(train_features, training=False)
shape_pred = tf.exp(shape_pred)


features = pd.DataFrame({"x1": [x1],
                         "x2": [x2],
                         "x3": [x3]})
colnames = features.columns.values.tolist()

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

ax1.scatter(train_features[:, 0], train_target, color="cornflowerblue", alpha=0.5, s=0.5)
ax1.scatter(train_features[:, 0], shape_pred[0], color="green", alpha=0.7, s=1.5)
ax1.scatter(train_features[:, 0], loc_pred[0], color="crimson", s=3.5)
ax1.set_xlabel(colnames[0])
ax1.set_ylabel("y")

ax2.scatter(train_features[:, 1], train_target, color="cornflowerblue", alpha=0.5, s=0.5)
ax2.scatter(train_features[:, 1], shape_pred[1], color="green", alpha=0.7, s=1.5)
ax2.scatter(train_features[:, 1], loc_pred[1], color="crimson", s=3.5)
ax2.set_xlabel(colnames[1])
ax2.set_ylabel("y")

ax3.scatter(train_features[:, 2], train_target, color="cornflowerblue", alpha=0.5, s=0.5)
ax3.scatter(train_features[:, 2], shape_pred[2], color="green", alpha=0.7, s=1.5)
ax3.scatter(train_features[:, 2], loc_pred[2], color="crimson", s=3.5)
ax3.set_xlabel(colnames[2])
ax3.set_ylabel("y")

plt.tight_layout(pad=0.4, w_pad=0.3)
plt.show()
