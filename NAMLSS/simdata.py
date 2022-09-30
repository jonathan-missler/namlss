import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import matplotlib.pyplot as plt


def feature_sim(seed=1866):
    dist1 = tfp.distributions.Uniform(-1, 1)
    x1 = dist1.sample((10000, 1), seed=seed)

    dist2 = tfp.distributions.Uniform()
    x2 = dist2.sample((10000, 1), seed=seed+1)

    dist3 = tfp.distributions.Uniform(1, 2)
    x3 = dist3.sample((10000, 1), seed=seed+2)

    return x1, x2, x3


def sim_gauss(seed=1866):
    x1, x2, x3 = feature_sim()

    ydist = tfp.distributions.Normal(loc=3*x1**2+-10*x2+5*tf.sin(4*x3),
                                     scale=0.5*(tf.exp(3*x1)+15*x2+1/x3))
    y = ydist.sample(seed=seed+3)

    return y, x1, x2, x3


def sim_gamma(seed=1866):
    x1, x2, x3 = feature_sim()

    ydist = tfp.distributions.Gamma(concentration=4*tf.abs(x1)+tf.abs(tf.sin(x2))+tf.exp(x3),
                                    rate=(tf.exp(3*x1)+15*x2+1/x3))
    y = ydist.sample(seed=seed+3)

    return y, x1, x2, x3


def sim_invgauss(seed=1866):
    x1, x2, x3 = feature_sim()

    ydist = tfp.distributions.InverseGaussian(loc=tf.cos(x1)**2+4*x2**3+tf.sqrt(x3),
                                              concentration=4*tf.abs(x1)+tf.abs(tf.sin(x2))+tf.exp(x3))
    y = ydist.sample(seed=seed + 3)

    return y, x1, x2, x3


y, x1, x2, x3 = sim_invgauss()

features = pd.DataFrame({"x1": [x1],
                         "x2": [x2],
                         "x3": [x3]})
colnames = features.columns.values.tolist()

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

ax1.scatter(x1, y, color="cornflowerblue", alpha=0.5, s=0.5)
ax1.set_xlabel(colnames[0])
ax1.set_ylabel("y")

ax2.scatter(x2, y, color="cornflowerblue", alpha=0.5, s=0.5)
ax2.set_xlabel(colnames[1])
ax2.set_ylabel("y")

ax3.scatter(x3, y, color="cornflowerblue", alpha=0.5, s=0.5)
ax3.set_xlabel(colnames[2])
ax3.set_ylabel("y")

plt.tight_layout(pad=0.4, w_pad=0.3)
plt.show()
