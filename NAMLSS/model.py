import tensorflow as tf
from NAMLSS.families import Gaussian
from neural_additive_models import models


class NamLSS(tf.keras.Model):

    def __init__(self,
                 num_inputs,
                 num_units,
                 trainable=True,
                 shallow=True,
                 family=Gaussian(two_param=True),
                 feature_dropout=0.0,
                 dropout=0.0,
                 **kwargs):

        super(NamLSS, self).__init__()
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

    def build(self, input_shape):
        self.mod1 = models.NAM(num_inputs=self._num_inputs, num_units=self._num_units, trainable=self._trainable,
                               shallow=self._shallow, feature_dropout=self._feature_dropout, dropout=self._dropout,
                               **self._kwargs)

        if self._family._two_param:
            self.mod2 = models.NAM(num_inputs=self._num_inputs, num_units=self._num_units, trainable=self._trainable,
                                shallow=self._shallow, feature_dropout=self._feature_dropout, dropout=self._dropout,
                                **self._kwargs)

        self.mod1.build(input_shape=input_shape)

        if self._family._two_param:
            self.mod2.build(input_shape=input_shape)

    def call(self, x, training=True):
        out1 = self.mod1(x, training=training)

        if self._family._two_param:
            out2 = self.mod2(x, training=training)
        else:
            out2 = 1

        return out1, out2

    def _name_scope(self):
        name_scope1 = self.mod1._namescope()

        if self._family._two_param:
            name_scope2 = self.mod2._namescope()

        return name_scope1, name_scope2

    def calc_outputs(self, x, training=True):
        outputs1 = self.mod1.calc_outputs(x, training=training)

        if self._family._two_param:
            outputs2 = self.mod2.calc_outputs(x, training=training)
        else:
            outputs2 = 1

        return outputs1, outputs2
