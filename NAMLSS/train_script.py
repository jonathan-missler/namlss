from neural_additive_models.data_utils import load_dataset, split_training_dataset
import tensorflow as tf
import numpy as np
from NAMLSS.model import NamLSS
from NAMLSS.trainer import Trainer
from NAMLSS.families import Gaussian
from NAMLSS.config import defaults

# define the config
config = defaults()
config.batch_size = 1024
config.activation = "exu"

# load and prepare data
features, target, _ = load_dataset("Housing")

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
config.num_epochs = 50
config.lr = 0.001
config.shallow = False
config.dropout = 0.1
config.feature_dropout = 0.1


family = Gaussian(two_param=True)
model = NamLSS(num_inputs=num_inputs, num_units=num_units, family=family, feature_dropout=config.feature_dropout,
               dropout=config.dropout, shallow=config.shallow, activation=config.activation)
optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)
trainer = Trainer(model, family, optimizer, config)

train_losses, val_losses = trainer.run_training(train_batches, val_batches)
