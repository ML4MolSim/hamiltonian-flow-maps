import flax.linen as nn
from hfm.backbones.embedding import GaussianRandomFourierFeatures
from hfm.backbones.utils import MLP
import jax.numpy as jnp


class MLPBackboneScalar(nn.Module):
    num_features: int = 256
    num_layers: int = 3
    activation_fn: str = 'silu'
    predict_scalar: bool = False

    @nn.compact
    def __call__(self, x):
        # x: (batch_size, num_nodes, num_dimensions)
        x = x.reshape(x.shape[0], -1)

        # embed x
        x_embed = MLP(
            num_layers=2,
            num_features=self.num_features,
            activation_fn=self.activation_fn,
            use_bias=True
        )(x)

        embed = nn.LayerNorm()(x_embed)
        embed = MLP(
            num_layers=self.num_layers,
            num_features=self.num_features,
            activation_fn=self.activation_fn,
            use_bias=True
        )(embed)

        embed = nn.LayerNorm()(embed)
        return nn.Dense(1)(embed)

    @nn.compact
    def __callx__(self, t, x, p):
        # t: (batch_size, 1)
        # x: (batch_size, num_nodes, num_dimensions)
        # p: (batch_size, num_nodes, num_dimensions)

        x = x.reshape(x.shape[0], -1)
        p = p.reshape(p.shape[0], -1)

        # embed time
        t_embed = GaussianRandomFourierFeatures(
            features=self.num_features_fourier
        )(t)
        t_embed = MLP(
            num_layers=2,
            num_features=self.num_features,
            activation_fn=self.activation_fn,
            use_bias=True
        )(t_embed)

        # embed x
        # xpt_embed = MLP(
        #     num_layers=2,
        #     num_features=self.num_features,
        #     activation=self.activation_fn,
        #     use_bias=False
        # )(jnp.concat([x, p, t], axis=-1))

        # embed p
        p_embed = MLP(
            num_layers=2,
            num_features=self.num_features,
            activation_fn=self.activation_fn,
            use_bias=True
        )(p)

        # embed p
        x_embed = MLP(
            num_layers=2,
            num_features=self.num_features,
            activation_fn=self.activation_fn,
            use_bias=True
        )(x)

        # embed = xpt_embed
        #embed = jnp.concat([t, x, p], axis=-1)
        #embed = jnp.concat([t_embed, x_embed, p_embed], axis=-1)  # (batch_size, 1 + dimX + dimP)
        embed = x_embed + p_embed + t_embed

        embed = MLP(
            num_layers=self.num_layers + 1,
            num_features=[self.num_features] * self.num_layers + [x.shape[-1] + p.shape[-1]],
            activation_fn=self.activation_fn,
            use_bias=True
        )(embed)

        # print(embed.shape)
        # print(f"embed: {embed.shape}, x: {x.shape}, p: {p.shape}")

        mean_v, mean_f = jnp.split(embed, [x.shape[-1]], axis=-1)
        return mean_v, mean_f
