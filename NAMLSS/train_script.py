from neural_additive_models.data_utils import load_dataset, split_training_dataset
import tensorflow as tf
import sklearn
import pandas as pd
import numpy as np
from NAMLSS.model import NamLSS
from NAMLSS.trainer import Trainer
from NAMLSS.families import Gaussian
from NAMLSS.config import defaults

config = defaults()

features, target, _ = load_dataset("Housing")

train_features, train_target, val_features, val_target = split_training_dataset(features, target, n_splits=1,
                                                                                stratified=False, random_state=1866)
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

family = Gaussian(two_param=True)
model = NamLSS()