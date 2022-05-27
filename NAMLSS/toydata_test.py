import tensorflow as tf
import tensorflow_probability as tfp
from NAMLSS.model import NamLSS
from NAMLSS.trainer import Trainer
from NAMLSS.config import defaults
from NAMLSS.families import Gaussian
import numpy as np
from neural_additive_models.data_utils import split_training_dataset
import matplotlib.pyplot as plt

config = defaults()
config.batch_size = 1000

Xdist = tfp.distributions.Normal(loc=0, scale=1)
x = Xdist.sample((1000, 1))

Ydist = tfp.distributions.Normal(loc=x, scale=x)
y = Ydist.sample()

data_array = np.array([x.numpy(), y.numpy()])
data_array = data_array.reshape((1000, 2))

split_generator = split_training_dataset(x.numpy(), y.numpy(), n_splits=1, stratified=False)

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
config.num_epochs = 100
config.lr = 0.01

family = Gaussian()
model = NamLSS(num_inputs=num_inputs, num_units=num_units, family=family, feature_dropout=config.feature_dropout,
               dropout=config.dropout, shallow=config.shallow, activation=config.activation)
optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)
trainer = Trainer(model, family, optimizer, config)

train_losses, val_losses = trainer.run_training(train_batches, val_batches)

loc_pred, scale_pred = trainer.model(x, training=False)
scale_pred = tf.exp(scale_pred)

plt.scatter(x.numpy(), y.numpy())
plt.plot(x.numpy(), loc_pred)
plt.plot(x.numpy(), loc_pred + 2*scale_pred)
plt.plot(x.numpy(), loc_pred - 2*scale_pred)
plt.show()