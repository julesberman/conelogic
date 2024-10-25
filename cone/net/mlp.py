from collections.abc import Iterable
from typing import Callable, List, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from cone.net.networks import get_activation, get_init


class MLP(nn.Module):
    features: List[int]
    activation: str = "swish"
    squeeze: bool = False
    use_bias: bool = True
    w_init: str = "lecun"

    @nn.compact
    def __call__(self, x):
        depth = len(self.features)
        w_init = get_init(self.w_init)

        A = get_activation(self.activation)
        for i, features in enumerate(self.features):
            is_last = i == depth - 1
            L = L = nn.Dense(features, use_bias=self.use_bias, kernel_init=w_init)
            x = L(x)
            if not is_last:
                x = A(x)

        if self.squeeze:
            x = jnp.squeeze(x)
        return x
