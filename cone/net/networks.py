import flax.linen as nn
import jax
import jax.numpy as jnp

from einops import rearrange
from flax.linen import initializers


import jax
import jax.numpy as jnp


def get_activation(activation):
    if activation == "relu":
        a = jax.nn.relu
    elif activation == "tanh":
        a = jax.nn.tanh
    elif activation == "sigmoid":
        a = jax.nn.sigmoid
    elif activation == "elu":
        a = jax.nn.elu
    elif activation == "selu":
        a = jax.nn.selu
    elif activation == "swish":
        a = jax.nn.swish
    elif activation == "sin":
        a = jnp.sin
    elif activation == "hswish":
        a = jax.nn.hard_swish

    return a


def get_init(init):

    if init is None or init == "lecun":
        w = initializers.lecun_normal()
    elif init == "ortho":
        w = initializers.orthogonal()
    elif init == "he":
        w = initializers.he_normal()
    return w
