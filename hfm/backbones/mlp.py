import flax.linen as nn
import jax
from hfm.backbones.embedding import SimpleTimeEmbedding
from hfm.backbones.utils import MLP, static_batch_jraph, static_unbatch_jraph
import jax.numpy as jnp

from hfm.jraph_utils import get_number_of_graphs


class MLPBackbone(nn.Module):
    num_features: int = 256
    num_layers: int = 3
    activation_fn: str = 'silu'
    dropout_rate: float = 0.0
    predict_grad_pot: bool = False
    name: str = "mlp"
    model_version: str = "0.0.1"

    def _pred(self, t, x, p, deterministic):
        # embed time
        time_embed = SimpleTimeEmbedding(
            num_features=self.num_features,
            activation_fn=self.activation_fn
        )(t)

        # embed x
        x_embed = MLP(
            num_layers=2,
            num_features=self.num_features,
            activation_fn=self.activation_fn,
            use_bias=True
        )(x)

        # embed p
        p_embed = MLP(
            num_layers=2,
            num_features=self.num_features,
            activation_fn=self.activation_fn,
            use_bias=True
        )(p)

        embed = x_embed + p_embed + time_embed
        embed = nn.LayerNorm()(embed)

        # main MLP block
        embed = MLP(
            num_layers=self.num_layers,
            num_features=self.num_features,
            activation_fn=self.activation_fn,
            use_bias=True,
            dropout_rate=self.dropout_rate
        )(embed, deterministic=deterministic)

        mean_Etot = MLP(
            num_layers=2,
            num_features=(self.num_features, 1),
            activation_fn=self.activation_fn,
            use_bias=True
        )(embed)

        if self.predict_grad_pot:
            # predict a single scalar (total energy)
            # aggregate over batches to compute gradients efficiently
            return jnp.sum(mean_Etot), mean_Etot

        else:
            # predict mean force and velocity directly
            # predict total energy as a scalar

            mean_f = MLP(
                num_layers=2,
                num_features=(self.num_features, x.shape[-1]),
                activation_fn=self.activation_fn,
                use_bias=True
            )(embed)

            mean_v = MLP(
                num_layers=2,
                num_features=(self.num_features, x.shape[-1]),
                activation_fn=self.activation_fn,
                use_bias=True
            )(embed)

            return mean_v, mean_f, mean_Etot

    def _internal_pred_grad(
            self,
            t, x, p,
            deterministic: bool = False
    ):
        energy_and_forces = jax.value_and_grad(
            self._pred, has_aux=True, argnums=(0, 1, 2)
        )
        (_, energy), (dEdt, forces, velocities) = energy_and_forces(t, x, p, deterministic)
        forces = -forces  # grad U = -F
        return velocities, forces, energy

    @nn.compact
    def __call__(self, t, sample, deterministic: bool = False):
        # t: (batch_size, 1)
        # x: (batch_size, num_nodes, num_dimensions)
        # p: (batch_size, num_nodes, num_dimensions)
        num_graphs = get_number_of_graphs(sample) - 1  # exclude padding graph

        org_shape = static_unbatch_jraph(num_graphs, sample.nodes["x"]).shape
        # flatten x, p
        x = static_unbatch_jraph(num_graphs, sample.nodes["x"]).reshape(num_graphs, -1)
        p = static_unbatch_jraph(num_graphs, sample.nodes["p"]).reshape(num_graphs, -1)
        t = t[:num_graphs]

        if self.predict_grad_pot:
            mean_v, mean_f, energy = self._internal_pred_grad(
                t, x, p,
                deterministic=deterministic
            )
        else:
            mean_v, mean_f, energy = self._pred(t, x, p, deterministic)
        
        mean_v, mean_f = mean_v.reshape(org_shape), mean_f.reshape(org_shape)
        pad_global = jnp.zeros((1, energy.shape[-1]))
        return static_batch_jraph(mean_v), static_batch_jraph(mean_f), jnp.concatenate([energy, pad_global], axis=0)
