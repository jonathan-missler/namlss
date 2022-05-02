import tensorflow as tf
from NAMLSS.families import Gaussian


class NAMLSS(tf.keras.module):


    def __init__(self,
                 num_inputs,
                 num_units,
                 trainable=True,
                 shallow=True,
                 family=Gaussian(two_param=True),
                 feature_dropout=0.0,
                 dropout=0.0,
                 **kwargs):
        super(namlss, self).__init__()
        self._num_inputs = num_inputs
        if isinstance(num_units, list):
            assert len(num_units) == num_inputs
            self._num_units = num_units
        elif isinstance(num_units, int):
            self._num_units = [num_units for _ in range(self._num_inputs)]
        self._trainable = trainable
        self._shallow = shallow
        self._family = family
        self._feature_dropout = feature_dropout
        self._dropout = dropout
        self._kwargs = kwargs
