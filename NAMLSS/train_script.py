from neural_additive_models.data_utils import load_dataset, split_training_dataset
import tensorflow as tf
import numpy as np
from NAMLSS.model import NamLSS
from NAMLSS.trainer import Trainer
from NAMLSS.families import Gaussian
from NAMLSS.config import defaults
import matplotlib.pyplot as plt

# define the config
config = defaults()
config.batch_size = 5024
config.activation = "relu"

# load and prepare data
features, target, colnames = load_dataset("Housing")

split_generator = split_training_dataset(features, target, n_splits=1, stratified=False, random_state=1245)

for i in split_generator:
    (train_features, train_target), (val_features, val_target) = i

train_data = tf.data.Dataset.from_tensor_slices((train_features, train_target))
val_data = tf.data.Dataset.from_tensor_slices((val_features, val_target))

train_batches = train_data.shuffle(1000).batch(config.batch_size)
val_batches = val_data.shuffle(1000).batch(config.batch_size)


# get num_inputs and num_units for NAM
num_unique_vals = [
    len(np.unique(train_features[:, i])) for i in range(train_features.shape[1])
]
num_units = [
    min(config.num_basis_functions, i * config.units_multiplier) for i in num_unique_vals
]
num_inputs = train_features.shape[-1]


# build objects for training
config.num_epochs = 300
config.lr = 0.001
config.shallow = False  #false

config.dropout = 0.1
config.feature_dropout = 0.1
config.early_stopping_patience = 10
config.output_regularization1 = 0.1 #0.1
config.output_regularization2 = 0.01 #0.01
config.l2_regularization1 = 0.1 #1.1
config.l2_regularization2 = 0.01 #0.01

family = Gaussian(two_param=True)
model = NamLSS(num_inputs=num_inputs, num_units=num_units, family=family, feature_dropout=config.feature_dropout,
               dropout=config.dropout, shallow=config.shallow, activation=config.activation)
optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)
trainer = Trainer(model, family, optimizer, config)

train_losses, val_losses = trainer.run_training(train_batches, val_batches)

loc_pred = trainer.model.mod1.calc_outputs(train_features, training=False)
scale_pred = trainer.model.mod2.calc_outputs(train_features, training=False)
scale_pred = tf.exp(scale_pred)

fig, ax = plt.subplots(nrows=2, ncols=4)
i = 0
for row in ax:
    for col in row:
        col.scatter(train_features[:, i], train_target, color="cornflowerblue", alpha=0.5, s=0.5)
        col.scatter(train_features[:, i], loc_pred[i] + 2*scale_pred[i], color="green", alpha=0.7, s=1.5)
        col.scatter(train_features[:, i], loc_pred[i] - 2*scale_pred[i], color="green", alpha=0.7, s=1.5)
        col.scatter(train_features[:, i], loc_pred[i], color="crimson", s=3.5)
        col.set_xlabel(colnames[i])
        col.set_ylabel("log(Price)")
        i += 1
plt.tight_layout(pad=0.4, w_pad=0.3)
plt.show()
