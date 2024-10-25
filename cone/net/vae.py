import flax.linen as nn
import jax
import jax.numpy as jnp

from einops import rearrange
from cone.net.networks import get_activation

import jax
import jax.numpy as jnp
from flax import linen as nn


class Encoder(nn.Module):
    latent_features: int
    features: list[int]
    kernel_size: int
    padding: str = "SAME"
    activation: str = "swish"

    @nn.compact
    def __call__(self, x):
        A = get_activation(self.activation)
        d_feats = self.features[-1]
        c_feats = self.features[:-1]

        for _, ff in enumerate(c_feats):
            x = nn.Conv(
                features=ff,
                kernel_size=(self.kernel_size, self.kernel_size),
                padding=self.padding,
            )(x)

        shape = x.shape  # Save shape for decoder
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=d_feats)(x)
        x = A(x)
        mean = nn.Dense(features=self.latent_features)(x)
        logvar = nn.Dense(features=self.latent_features)(x)
        return mean, logvar, shape


class Decoder(nn.Module):
    latent_features: int
    features: list[int]
    kernel_size: int
    padding: str = "SAME"
    activation: str = "swish"

    @nn.compact
    def __call__(self, z, shape):
        A = get_activation(self.activation)

        d_feats = self.features[0]
        c_feats = self.features[1:]
        x = nn.Dense(features=d_feats)(z)
        x = A(x)
        x = nn.Dense(features=shape[1] * shape[2] * shape[3])(x)
        x = A(x)
        x = x.reshape((x.shape[0], shape[1], shape[2], shape[3]))

        for i, ff in enumerate(c_feats):
            x = nn.ConvTranspose(
                features=ff,
                kernel_size=(self.kernel_size, self.kernel_size),
                padding=self.padding,
            )(x)

            if i < len(c_feats) - 1:
                x = A(x)

        return x


class VAE(nn.Module):
    latent_features: int
    encoder_features: list[int]
    decoder_features: list[int]
    kernel_size: int
    padding: str = "SAME"
    activation: str = "swish"

    def setup(self):
        self.encoder = Encoder(
            self.latent_features,
            self.encoder_features,
            self.kernel_size,
            self.padding,
            activation=self.activation,
        )
        self.decoder = Decoder(
            self.latent_features,
            self.decoder_features,
            self.kernel_size,
            self.padding,
            activation=self.activation,
        )

    def __call__(self, x):
        mean, logvar, shape = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decode(z, shape)
        return recon_x, mean, logvar

    def encode(self, x):
        mean, logvar, shape = self.encoder(x)
        return mean, logvar, shape

    def decode(self, z, shape):
        recon_x = self.decoder(z, shape)
        return recon_x

    def encode_latent(self, x, key):
        mean, logvar, _ = self.encoder(x)
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, std.shape)
        return mean + eps * std

    def reparameterize(self, mean, logvar):
        std = jnp.exp(0.5 * logvar)
        rng = self.make_rng("latent")
        eps = jax.random.normal(rng, std.shape)
        return mean + eps * std
