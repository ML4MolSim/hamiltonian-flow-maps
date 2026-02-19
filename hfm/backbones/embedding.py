from typing import Optional

import e3x
from hfm import jraph_utils
from hfm.backbones.base import BaseEdgeEmbedding, BaseNodeEmbedding, BaseTimeEmbedding
from hfm.backbones.utils import MLP, get_cutoff_value, get_e3x_radial_basis_fn, promote_to_e3x, soft_gaussian, E3MLP
import jax.numpy as jnp
import jax
import flax.linen as nn
import functools


class GaussianRandomFourierFeatures(nn.Module):
    features: int
    sigma: float = 1.
    dtype = jnp.float32
    param_dtype = jnp.float32

    @nn.compact
    def __call__(
            self,
            x  # (..., d)
    ):
        if self.features % 2 != 0:
            raise ValueError(
                f'features must be even. '
                f'received {self.features=}'
            )

        b = self.param(
            'b',
            jax.nn.initializers.normal(self.sigma),
            (x.shape[-1], self.features // 2),
            self.param_dtype
        )  # (d, features // 2)

        bT_x = jnp.einsum(
            '...d, dh -> ...h',
            x,
            b
        )  # (..., features // 2)

        cos = jnp.cos(2 * jnp.pi * bT_x)  # (..., features // 2)
        sin = jnp.sin(2 * jnp.pi * bT_x)  # (..., features // 2)

        # gamma contains alternating cos and sin terms by first stacking and then reshaping.
        gamma = jnp.stack(
            [cos, sin],
            axis=-1
        ).reshape(*cos.shape[:-1], -1)  # (..., features)

        return gamma


class SimpleTimeEmbedding(nn.Module):
    num_features: int
    num_features_fourier: int = None
    activation_fn: str = 'silu'

    @nn.compact
    def __call__(self, time_latent, *args, **kwargs):
        num_features_fourier = self.num_features // 2 if self.num_features_fourier is None else self.num_features_fourier

        ff = GaussianRandomFourierFeatures(
            features=num_features_fourier
        )(
            time_latent
        )

        features_time = MLP(
            num_layers=2,
            activation_fn=self.activation_fn,
            num_features=self.num_features,
            use_bias=True
        )(
            ff
        )

        return features_time


class TimeEmbedding(BaseTimeEmbedding):
    num_features: int
    num_features_fourier: int = None
    activation_fn: str = 'silu'

    @nn.compact
    def __call__(self, time_latent, *args, **kwargs):

        num_features_fourier = self.num_features // 2 if self.num_features_fourier is None else self.num_features_fourier

        ff = GaussianRandomFourierFeatures(
            features=num_features_fourier
        )(
            jnp.expand_dims(time_latent, axis=-1)
        )  # (num_nodes, num_features)

        features_time = MLP(
            num_layers=2,
            activation_fn=self.activation_fn,
            num_features=self.num_features,
            use_bias=True
        )(
            ff
        )  # (num_nodes, 1, 1, num_features)

        features_time = promote_to_e3x(features_time)  # (num_nodes, 1, 1, num_features)

        return features_time

    
class TimeEmbeddingV2(BaseTimeEmbedding):
    num_features: int
    num_features_fourier: int = None
    activation_fn: str = 'silu'
    use_mass_cond: bool = False
    use_velocity_cond: bool = False
    velocities_num_basis: int = 8
    velocities_max_value: float = 1.55

    @nn.compact
    def __call__(self, time_latent, *args, **kwargs):

        graph = kwargs['graph']
        num_features_fourier = self.num_features // 2 if self.num_features_fourier is None else self.num_features_fourier

        ff = GaussianRandomFourierFeatures(
            features=num_features_fourier
        )(
            jnp.expand_dims(time_latent, axis=-1)
        )  # (num_nodes, num_features)

        features_time = MLP(
            num_layers=2,
            activation_fn=self.activation_fn,
            num_features=self.num_features,
            use_bias=True
        )(
            ff
        )  # (num_nodes, 1, 1, num_features)

        if self.use_mass_cond:
            masses = graph.nodes['masses']  # (num_nodes, 1)
            features_masses = MLP(
                num_layers=2,
                num_features=self.num_features,
                activation_fn=self.activation_fn
            )(
                masses
            )  # (num_nodes, num_features)

            features_time = jnp.concat([features_time, features_masses], axis=-1) # (num_nodes, 2*num_features)

        if self.use_velocity_cond:
            velocities = graph.nodes['p'] / graph.nodes['masses']  # (num_nodes, 3)
            atomic_numbers = graph.nodes['atomic_numbers'].reshape(-1) # (num_nodes, 1)

            velocities_norm = e3x.ops.norm(velocities, axis=-1) # (num_nodes)

            rbf = soft_gaussian(
                velocities_norm,
                num=self.velocities_num_basis,
                limit=self.velocities_max_value,
            ) # (num_nodes, num_basis)

            atom_onehot = nn.one_hot(atomic_numbers, num_classes=119)
            W = nn.Dense(
                self.velocities_num_basis * self.velocities_num_basis, 
                use_bias=False
            )(
                atom_onehot
            ).reshape(-1, self.velocities_num_basis, self.velocities_num_basis)
            rbf = jnp.einsum("nb,nbc->nc", rbf, W) # (num_nodes, num_basis)

            y = jnp.concat([rbf, velocities], axis=-1) # (num_nodes, num_basis+3)

            features_velocities = MLP(
                num_layers=2,
                num_features=self.num_features,
                activation_fn=self.activation_fn
            )(
                y
            ) # (num_nodes, num_features)

            features_time = jnp.concat([features_time, features_velocities], axis=-1) # (num_nodes, 2*num_features) or (num_nodes, 3*num_features)

        if self.use_mass_cond or self.use_velocity_cond:
            features_time = MLP(
                num_layers=2,
                num_features=self.num_features,
                activation_fn=self.activation_fn
            )(
                features_time
            )  # (num_nodes, num_features)

        features_time = promote_to_e3x(features_time)  # (num_nodes, 1, 1, num_features)

        return features_time
    
    
class DiTNodeEmbed(BaseNodeEmbedding):
    num_features: int
    activation_fn: str
    
    mass_embedding_bool: bool = False
    positional_embedding_bool: bool = False
    
    velocities_embedding_bool: bool = True
    velocities_encode_magnitude_bool: bool = True
    velocities_num_basis: int = 8
    velocities_max_frequency: float = 4*jnp.pi

    @nn.compact
    def __call__(self, graph, **kwargs):
        atomic_numbers = graph.nodes['atomic_numbers'].reshape(-1)

        h = nn.Embed(
            features=self.num_features,
            num_embeddings=119
        )(
            atomic_numbers
        )  # (num_nodes, num_features)

        # Mass embedding.
        if self.mass_embedding_bool:
            masses = graph.nodes['masses']  # (num_nodes, 1)
            masses_embed = MLP(
                num_features=self.num_features,
                num_layers=2,
                activation_fn=self.activation_fn
            )(
                masses
            )  # (num_nodes, num_features)
            h += masses_embed  # (num_nodes, num_features)
        
        # Absolute positional embedding.
        if self.positional_embedding_bool:
            positions = graph.nodes['x']  # (num_nodes, 3)
            x_embed = MLP(
                num_features=self.num_features,
                num_layers=2,
                activation_fn=self.activation_fn
            )(
                positions
            )  # (num_nodes, num_features)
            h += x_embed  # (num_nodes, num_features)

        if self.velocities_embedding_bool:
            
            velocities = graph.nodes['p'] / graph.nodes['masses']  # (num_nodes, 3)
            
            # ensure velocity doesnt contain NaNs before passing it on
            node_mask = jraph_utils.get_node_padding_mask(graph)
            velocities = jnp.where(node_mask[:, None], velocities, 0.)

            v_embed = VelocityEmbedding(
                num_features=self.num_features,
                encode_magnitude_bool=self.velocities_encode_magnitude_bool,
                num_basis=self.velocities_num_basis,
                max_frequency=self.velocities_max_frequency,
                activation_fn=self.activation_fn
            )(
                velocities=velocities,
                atomic_numbers=atomic_numbers
            ) # (num_nodes, num_features)

            h += v_embed  # (num_nodes, num_features)

        return promote_to_e3x(h)  # (num_nodes, 1, 1, num_features)


class VelocityEmbedding(nn.Module):
    num_features: int
    activation_fn: str
    
    encode_magnitude_bool: bool = True
    num_basis: Optional[int] = None
    max_frequency: Optional[float] = None

    def setup(self):
        if self.encode_magnitude_bool:
            if self.num_basis is None or self.max_frequency is None:
                raise ValueError(f'For magnitude encoding `num_basis` and `max_frequency` must be set. Received {self.num_basis=} and {self.max_frequency=}')

    @nn.compact
    def __call__(self, velocities, atomic_numbers, *args, **kwargs):

        if self.encode_magnitude_bool:    
            velocities_norm = e3x.ops.norm(velocities, axis=-1) # (num_nodes)
            rbf = e3x.nn.basic_fourier(
                velocities_norm,
                num=self.num_basis,
                limit=(self.num_basis - 1) * jnp.pi / self.max_frequency
            ) # (num_nodes, num_basis)

            atom_onehot = nn.one_hot(atomic_numbers, num_classes=119)
            W = nn.Dense(self.num_basis * self.num_basis, use_bias=False)(atom_onehot).reshape(-1, self.num_basis, self.num_basis)
            rbf = jnp.einsum("nb,nbc->nc", rbf, W) # (num_nodes, num_basis)

            y = jnp.concat([rbf, velocities], axis=-1) # (num_nodes, num_basis+3)
        else:
            y = velocities # (num_nodes, 3)

        velocity_embedding = MLP(
            num_layers=2,
            num_features=self.num_features,
            activation_fn=self.activation_fn
        )(
            y
        ) # (num_nodes, num_features)

        return velocity_embedding # (num_nodes, num_features)


class DiTEdgeEmbed(BaseEdgeEmbedding):
    num_features: int
    activation_fn: str

    cutoff: float
    cutoff_fn: str = 'smooth_cutoff'
    
    radial_basis: str = 'basic_fourier'
    num_basis: int = 10

    embed_rel_velocities_bool: bool = False
    model_version: str = "0.0.1"

    @nn.compact
    def __call__(self, graph, **kwargs):

        num_edges = len(graph.senders)

        # default value
        re = jnp.zeros(
                (num_edges, self.num_features),
                dtype=graph.nodes['x'].dtype
            )
        
        positions = graph.nodes['x']  # (num_nodes, 3)
        # ensure velocipositionsty doesnt contain NaNs before passing it on
        node_mask = jraph_utils.get_node_padding_mask(graph)
        positions = jnp.where(node_mask[:, None], positions, 0.)

        src_idx = graph.senders  # (num_edges,)
        dst_idx = graph.receivers  # (num_edges,)

        # Calculate the displacements and normalized directions.
        displacements = positions[src_idx] - positions[dst_idx]  # (num_pairs, 3)
        directions, distances = e3x.ops.normalize_and_return_norm(
            displacements, 
            axis=-1, 
            keepdims=False
        )  # (num_pairs, 3), (num_pairs, )

        radial_basis_fn = get_e3x_radial_basis_fn(self.radial_basis, cutoff=self.cutoff)
            
        edge_value = radial_basis_fn(
            distances,
            num=self.num_basis,
        )  # (num_pairs, num_basis)

        edge_value = jnp.concat([edge_value, directions], axis=-1)  # (num_pairs, num_basis + 3)
    
        edge_embedding = MLP(
            num_features=self.num_features,
            num_layers=2,
            use_bias=True,
            activation_fn=self.activation_fn
        )(
            edge_value
        )  # (num_pairs, num_features)

        re += edge_embedding  # (num_edges, num_features)

        if self.embed_rel_velocities_bool:
            velocities = graph.nodes['p'] / graph.nodes['masses']  # (num_nodes, 3)

            # ensure velocity doesnt contain NaNs before passing it on
            node_mask = jraph_utils.get_node_padding_mask(graph)
            velocities = jnp.where(node_mask[:, None], velocities, 0.)

            # Calculate the displacements.
            rel_vel = velocities[src_idx] - velocities[dst_idx]  # (num_pairs, 3)

            rel_vel_embed = MLP(
                num_features=self.num_features,
                num_layers=2,
                use_bias=True,
                activation_fn=self.activation_fn
            )(
                rel_vel
            )  # (num_pairs, num_features)

            re += rel_vel_embed  # (num_edges, num_features)

        return promote_to_e3x(re)  # (num_edges, 1, 1, num_features)

### SO3 Equivariant Modules ###

class SO3EdgeEmbedding(BaseEdgeEmbedding):
    max_degree: int
    activation_fn: str

    cutoff: float
    cutoff_fn: str = 'smooth_cutoff'
    
    radial_basis: str = 'basic_fourier'
    num_basis: int = 10
    model_version: str = "0.0.1"

    @nn.compact
    def __call__(self, graph, **kwargs):
        src_idx = graph.senders
        dst_idx = graph.receivers

        positions = graph.nodes['x']

        # Calculate the displacements.
        displacements = positions[src_idx] - positions[dst_idx]  # (num_pairs, 3)

        radial_fn = get_e3x_radial_basis_fn(self.radial_basis, cutoff=self.cutoff)
        cutoff_fn = functools.partial(getattr(e3x.nn, self.cutoff_fn), cutoff=self.cutoff)
        cutoff_fn = None
        # cutoff will be applied later during self attention
        
        basis = e3x.nn.basis(
            displacements,
            num=self.num_basis,
            max_degree=self.max_degree,
            radial_fn=radial_fn,
            cutoff_fn=cutoff_fn
        )  # (num_pairs, 1, (max_degree+1)**2, num_basis_functions)

        return basis


class SO3NodeEmbed(BaseNodeEmbedding):
    num_features: int
    activation_fn: str
    
    mass_embedding_bool: bool = False
    
    velocities_embedding_bool: bool = True
    velocities_encode_magnitude_bool: bool = True
    velocities_num_basis: int = 8
    velocities_max_frequency: float = 4*jnp.pi

    @nn.compact
    def __call__(self, graph, **kwargs):
        atomic_numbers = graph.nodes['atomic_numbers'].reshape(-1)
    
        h = e3x.nn.Embed(
            features=self.num_features,
            num_embeddings=119
        )(
            atomic_numbers
        )  # (num_nodes, 1, 1, num_features)

        # Mass embedding.
        if self.mass_embedding_bool:
            masses_embed = E3MLP(
                num_layers=2,
                activation_fn=self.activation_fn,
                num_features=self.num_features
            )(
                graph.nodes['masses'][:, None, None]
            )  # (num_nodes, 1, 1, num_features)

            h = e3x.nn.add(h, masses_embed)  # (num_nodes, 1, 1, num_features)

        if self.velocities_embedding_bool:
            velocities = graph.nodes['p'] / graph.nodes['masses']  # (num_nodes, 3)

            # ensure velocity doesnt contain NaNs before passing it on
            node_mask = jraph_utils.get_node_padding_mask(graph)
            velocities = jnp.where(node_mask[:, None], velocities, 0.)

            v_embed = SO3VelocityEmbedding(
                num_features=self.num_features,
                encode_magnitude_bool=self.velocities_encode_magnitude_bool,
                num_basis=self.velocities_num_basis,
                max_frequency=self.velocities_max_frequency,
                activation_fn=self.activation_fn
            )(
                velocities=velocities
            ) # (num_nodes, 1, 4, num_features)

            h = e3x.nn.add(h, v_embed)  # (num_nodes, 1, 4, num_features)

        return h  # (num_nodes, 1, 4, num_features) or (num_nodes, 1, 1, num_features) without velocity_embedding.
    

class SO3VelocityEmbedding(nn.Module):
    num_features: int
    activation_fn: str

    encode_magnitude_bool: bool = True
    num_basis: Optional[int] = None
    max_frequency: Optional[float] = None

    def setup(self):
        if self.encode_magnitude_bool:
            if self.num_basis is None or self.max_frequency is None:
                raise ValueError(f'For magnitude encoding `num_basis` and `max_frequency` must be set. Received {self.num_basis=} and {self.max_frequency=}')

    @nn.compact
    def __call__(self, velocities, *args, **kwargs):
        """
        Args:
            velocities: (num_nodes, 3)
        """
        if self.encode_magnitude_bool:
            velocities_norm = e3x.ops.norm(velocities, axis=-1) # (num_nodes)
            rbf = e3x.nn.basic_fourier(
                velocities_norm,
                num=self.num_basis,
                limit=(self.num_basis - 1) * jnp.pi / self.max_frequency
            ) # (num_nodes, num_basis)
            rbf = rbf[:, None, None, :] # (num_nodes, 1, 1, num_basis)
        else:
            rbf = jnp.zeros((len(velocities), 1, 1, self.num_basis))

        y = nn.Dense(
            self.num_basis,
            use_bias=False
        )(
            velocities[:, None, :, None] # (num_nodes, 1, 3, num_basis)
        )

        y = jnp.concat([rbf, y], axis=-2) # (num_node, 1, 4, num_basis)

        if self.encode_magnitude_bool:
            velocity_embedding = E3MLP(
                num_layers=2,
                num_features=self.num_features,
                activation_fn=self.activation_fn
            )(
                y
            ) # (num_nodes, 1, 4, num_features)
        else:
            # The weights for the l=0 channel are not doing anything and are just multiplied with the zeros from above. 
            # Therefore we also do not use any bias, such that the l=0 channels will always be zero. The weights in the dense
            # layer will also always have zero gradient such that they won't be updated.
            velocity_embedding = e3x.nn.Dense(
                features=self.num_features,
                use_bias=False
            )(
                y
            )

        return velocity_embedding # (num_nodes, 1, 4, num_features)
