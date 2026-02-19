import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import logging
import numpy as np

from jaxtyping import Array
from typing import Optional

from hfm.backbones import embedding

from .base import BaseReadout
from .base import FeatureRepresentations
from .base import GenerativeLayer

from .dit import DiTModel

from .utils import MLP
from .utils import E3MLP
from .utils import EquivariantLayerNorm
from .utils import ZBLRepulsion

from .utils import broadcast_equivariant_multiplication
from .utils import get_activation_fn
from .utils import get_max_degree_from_tensor_e3x
from .utils import modulate_E3adaLN

from ..jraph_utils import get_batch_segments, get_number_of_graphs, get_number_of_nodes, get_node_padding_mask


class SO3DiTLayer(nn.Module):
    num_heads: int
    num_features_mlp: int
    max_degree: int
    include_pseudotensors: bool
    activation_fn_mlp: str = 'gelu'
    activation_fn: str = 'silu'
    use_adaptive_layernorm: bool = True
    relative_embedding_qk_bool: bool = True
    relative_embedding_v_bool: bool = True
    model_version: str = "0.0.1"

    @nn.compact
    def __call__(
            self,
            graph,
            features: FeatureRepresentations,
            **kwargs
    ):
        num_nodes = get_number_of_nodes(graph)

        src_idx = graph.senders
        dst_idx = graph.receivers

        cutoff_value = kwargs['cutoff_value']  # (num_pairs)
        features_time = kwargs['features_time']  # (num_nodes, 1, 1, num_features)
        features_nodes = features.nodes  # (num_nodes, P, (max_degree + 1)**2, num_features)
        features_edges = features.edges  # (num_nodes, P, (max_degree + 1)**2, num_features)

        num_features = features_nodes.shape[-1]

        assert features_nodes.ndim == 4, 'Features are assumed to be in the e3x convention.'

        if self.use_adaptive_layernorm:
            # Calculate the shift and scale parameters for adaLN and adaLN-Zero
            c = nn.LayerNorm()(features_time)

            max_degree_input = get_max_degree_from_tensor_e3x(features_nodes)

            parity_input = features_nodes.shape[-3]
            parity_output = 2 if self.include_pseudotensors else 1

            act_fn = get_activation_fn(self.activation_fn)
 
            c = nn.Dense(
                    features=num_features * (max_degree_input + 1) * parity_input + num_features * (
                        self.max_degree + 1) * parity_output + 2 * num_features + 2 * num_features * (
                        self.max_degree + 1) * parity_output,
                    kernel_init=jax.nn.initializers.zeros
            )(
                act_fn(c)
            )  # 6 times (num_nodes, num_features)

            gamma1, beta1, alpha1, gamma2, beta2, alpha2 = jnp.split(
                c,
                indices_or_sections=np.array(
                    [
                        num_features * (max_degree_input + 1) * parity_input,
                        num_features * (max_degree_input + 1) * parity_input + num_features,
                        num_features * (max_degree_input + 1) * parity_input + num_features + num_features * (
                                    self.max_degree + 1) * parity_output,
                        num_features * (max_degree_input + 1) * parity_input + num_features + num_features * (
                                self.max_degree + 1) * parity_output * 2,
                        num_features * (max_degree_input + 1) * parity_input + num_features + num_features * (
                                self.max_degree + 1) * parity_output * 2 + num_features
                    ]
                ),
                axis=-1
            )

            gamma1 = gamma1.reshape(num_nodes, parity_input, (max_degree_input + 1), num_features)
            beta1 = beta1.reshape(num_nodes, 1, 1, num_features)
            alpha1 = alpha1.reshape(num_nodes, parity_output, (self.max_degree + 1), num_features)
            gamma2 = gamma2.reshape(num_nodes, parity_output, (self.max_degree + 1), num_features)
            beta2 = beta2.reshape(num_nodes, 1, 1, num_features)
            alpha2 = alpha2.reshape(num_nodes, parity_output, (self.max_degree + 1), num_features)

            # SO3 equivariant adaLN.
            features_nodes_pre_attention = modulate_E3adaLN(
                x=EquivariantLayerNorm(use_bias=False, use_scale=False)(features_nodes),
                scale=gamma1,
                shift=beta1
            )
        else:
            features_nodes_pre_attention = EquivariantLayerNorm()(features_nodes)


        with jax.debug_nans(False):
            with jax.debug_infs(False):
                features_nodes_post_att = e3x.nn.SelfAttention(
                    num_heads=self.num_heads,
                    max_degree=self.max_degree,
                    use_fused_tensor=False,
                    include_pseudotensors=self.include_pseudotensors,
                    use_relative_positional_encoding_qk=self.relative_embedding_qk_bool,
                    use_relative_positional_encoding_v=self.relative_embedding_v_bool
                )(
                    features_nodes_pre_attention,
                    features_edges,
                    cutoff_value=cutoff_value,
                    dst_idx=dst_idx,
                    src_idx=src_idx,
                    num_segments=num_nodes
                )  # (num_nodes, parity, (max_degree + 1)**2, num_features)

                features_nodes_post_att = jnp.where(jraph.segment_mean(cutoff_value, dst_idx, num_nodes)\
                                                    .reshape(-1,1,1,1) < 1e-5, features_nodes_pre_attention, features_nodes_post_att)

        if self.use_adaptive_layernorm:
            # Skip connection with SO3 equivariant per-degree / parity scaling.
            features_nodes = e3x.nn.add(
                features_nodes,
                broadcast_equivariant_multiplication(
                    factor=alpha1,
                    tensor=features_nodes_post_att
                )
            )  # (num_nodes, parity, (max_degree + 1)**2, num_features)

            features_nodes_pre_mlp = modulate_E3adaLN(
                x=EquivariantLayerNorm(use_bias=False, use_scale=False)(features_nodes),
                scale=gamma2,
                shift=beta2
            )  # (num_nodes, parity, (max_degree + 1)**2, num_features)
        else:
            # Skip connection
            features_nodes = e3x.nn.add(
                features_nodes, 
                features_nodes_post_att
            )  # (num_nodes, parity, (max_degree + 1)**2, num_features)

            features_nodes_pre_mlp = EquivariantLayerNorm()(
                features_nodes
            )  # (num_nodes, parity, (max_degree + 1)**2, num_features)

        features_nodes_post_mlp = E3MLP(
            num_features=[self.num_features_mlp, num_features],
            num_layers=2,
            activation_fn=self.activation_fn_mlp,
        )(
            features_nodes_pre_mlp
        )  # (num_nodes, parity, (max_degree + 1)**2, num_features)

        if self.use_adaptive_layernorm:
            # Skip connection with scaling.
            features_nodes = e3x.nn.add(
                features_nodes,
                broadcast_equivariant_multiplication(
                    factor=alpha2,
                    tensor=features_nodes_post_mlp
                )
            )   # (num_nodes, parity, (max_degree + 1)**2, num_features)
        else:
            # Skip connection
            features_nodes = e3x.nn.add(
                features_nodes,
                features_nodes_post_mlp
            )   # (num_nodes, parity, (max_degree + 1)**2, num_features)

        return FeatureRepresentations(
            nodes=features_nodes,
            edges=features_edges
        )


class EquivariantReadout(BaseReadout):
    activation_fn: str
    use_adaptive_layernorm: bool = True
    zbl_repulsion_bool: bool = False
    cutoff: Optional[float] = None

    def setup(self):
        if self.zbl_repulsion_bool:
            if self.cutoff is None:
                raise ValueError(
                    f"ZBL repulsion requires cutoff to be set. received {self.zbl_repulsion_bool=} but {self.cutoff=}."
                )
    

    @nn.compact
    def __call__(
            self,
            graph: jraph.GraphsTuple,
            features: FeatureRepresentations,
            features_time: Array,
            *args,
            **kwargs
    ):

        features_nodes = features.nodes  # (num_nodes, 1, 1, num_features)
        assert features_nodes.ndim == 4, 'Features are assumed to be in the e3x convention.'

        num_features = features_nodes.shape[-1]
        num_nodes = len(features_nodes)

        y = e3x.nn.change_max_degree_or_type(features_nodes, max_degree=1, include_pseudotensors=False) # (num_nodes, 1, 4, num_features)

        if self.use_adaptive_layernorm:
            # Calculate the shift and scale parameters for adaLN and adaLN-Zero
            c = nn.LayerNorm()(features_time)  # (num_nodes, 1, 1, num_features)

            act_fn = get_activation_fn(self.activation_fn)

            scale, shift = jnp.split(
                nn.Dense(
                    features=3 * num_features,
                    kernel_init=jax.nn.initializers.zeros
                )(
                    act_fn(c)
                ),
                indices_or_sections=np.array(
                    [
                        2 * num_features,
                    ]
                ),
                axis=-1
            )

            scale = scale.reshape(num_nodes, 1, 2, num_features)
            shift = shift.reshape(num_nodes, 1, 1, num_features)

            y = modulate_E3adaLN(
                x=EquivariantLayerNorm(use_scale=False, use_bias=False)(y),
                scale=scale,
                shift=shift
            ) # (num_nodes, 1, 4, num_features)
        else:
            y = EquivariantLayerNorm()(
                y
            ) # (num_nodes, 1, 4, num_features)

        mean_v = E3MLP(
            num_layers=2,
            activation_fn=self.activation_fn,
            num_features=(num_features, 1)
        )(
            y
        )[:, 0, 1:, 0] # (num_nodes, 3)

        mean_f = E3MLP(
            num_layers=2,
            activation_fn=self.activation_fn,
            num_features=(num_features, 1)
        )(
            y
        )[:, 0, 1:, 0] # (num_nodes, 3)

        y = e3x.nn.change_max_degree_or_type(y, max_degree=0, include_pseudotensors=False) # (num_nodes, 1, 1, num_features)
        y = jnp.squeeze(y, axis=(1, 2))
        
        energy = MLP(
            num_layers=2,
            activation_fn=self.activation_fn,
            num_features=(num_features, 1),
            use_bias=True
        )(
            y
        ) # (num_nodes, 1)

        if self.zbl_repulsion_bool:
            energy_zbl = ZBLRepulsion(
                cutoff=self.cutoff
            )(
                graph=graph
            ) # (num_nodes, 1)

            energy += energy_zbl # (num_nodes, 1)
        
        # Mask out contribution from the padding nodes.
        node_mask = get_node_padding_mask(graph) # (num_nodes)
        energy = jnp.where(jnp.expand_dims(node_mask, axis=-1), energy, 0.0) # (num_nodes, 1)

        # sum energy contributions over atoms in each graph
        num_graphs = get_number_of_graphs(graph)
        batch_segments = get_batch_segments(graph)
        energy = jax.ops.segment_sum(energy, batch_segments, num_graphs) # (num_graphs, 1)

        return mean_v, mean_f, energy


def make_molecular_SO3DiT(
        num_layers: int,
        num_heads: int,
        num_features_head: int,
        max_degree: int,
        cutoff: float,
        include_pseudotensors: bool = True,
        num_features_mlp: int = None,
        use_adaptive_layernorm: bool = True,
        rpe_radial_basis: str = 'basic_fourier',
        rpe_num_basis: int = 10,
        activation_fn_mlp: str = 'gelu',
        activation_fn: str = 'silu',
        scale_pos: float = 1.0,
        scale_mom: float = 1.0,
        name: str = "so3dit",
        embed_positions_absolute: bool = False,
        embed_velocities_absolute: bool = True,
        embed_velocities_relative: bool = False,
        embed_masses: bool = False,
        velocities_embed_encode_magnitude: bool = False,
        velocities_embed_num_basis: int = 8,
        velocities_embed_max_frequency: float = 4*jnp.pi,
        force_as_grad: bool = False,
        zbl_repulsion: bool = False,
        model_version="0.0.1",
        **kwargs
):
    
    if "embed_positions_relative" in kwargs:
        raise DeprecationWarning(
            f'`embed_positions_relative` has been depracated, since this should be always done. At least for the moment.'
        )

    if embed_velocities_relative:
        raise NotImplemented(f'`embed_velocities_relative` is not implemented for DiTSO3 yet.')

    if embed_positions_absolute:
        raise NotImplemented(f'`embed_positions_absolute` is not implemented for DiTSO3 yet.')

    if zbl_repulsion:
        if force_as_grad == False:
            logging.warning(
                f"ZBL repulsion is only added to the final energy prediction. Thus, when forces are not obtained via the gradient, they generally missmatch with the energy when ZBL repuslion is enabled. "
                f"This is non-critical when using only the energy predictions of the model (i.e., for momentum re-scaling), but when running dynamics and / or training on energy and forces, "
                f"this can lead to a missmatch which should be avoided. Received {zbl_repulsion=} but {force_as_grad=}."
            )

    num_features = num_heads * num_features_head

    if num_features_mlp is None:
        num_features_mlp = 4 * num_features

    # Time embedding
    time_embedding = embedding.TimeEmbedding(
        num_features=num_features,
        activation_fn=activation_fn
    )

    # Node embedding
    node_embedding = embedding.SO3NodeEmbed(
        num_features=num_features,
        activation_fn=activation_fn,
        mass_embedding_bool=embed_masses,
        # positional_embedding_bool=embed_positions_absolute,
        velocities_embedding_bool=embed_velocities_absolute,
        velocities_encode_magnitude_bool=velocities_embed_encode_magnitude,
        velocities_num_basis=velocities_embed_num_basis,
        velocities_max_frequency=velocities_embed_max_frequency
    )

    # Edge embedding
    edge_embedding = embedding.SO3EdgeEmbedding(
        max_degree=max_degree,
        activation_fn=activation_fn,
        cutoff=cutoff,
        radial_basis=rpe_radial_basis,
        num_basis=rpe_num_basis,
        model_version=model_version,
        # embed_rel_velocities_bool=embed_velocities_relative
    )

    # Readout block
    readout_block = EquivariantReadout(
        activation_fn=activation_fn,
        use_adaptive_layernorm=use_adaptive_layernorm,
        zbl_repulsion_bool=zbl_repulsion,
        cutoff=cutoff
    )

    layers = []

    for _ in range(num_layers):
        layers.append(
            GenerativeLayer(
                encoder=SO3DiTLayer(
                    num_heads=num_heads,
                    num_features_mlp=num_features_mlp,
                    max_degree=max_degree,
                    include_pseudotensors=include_pseudotensors,
                    activation_fn_mlp=activation_fn_mlp,
                    activation_fn=activation_fn,
                    use_adaptive_layernorm=use_adaptive_layernorm,
                    relative_embedding_qk_bool=True,
                    relative_embedding_v_bool=True,
                    model_version=model_version
                )
            )
        )

    return DiTModel(
        time_embedding=time_embedding,
        node_embedding=node_embedding,
        edge_embedding=edge_embedding,
        layers=layers,
        readout=readout_block,
        name=name,
        scale_pos=scale_pos,
        scale_mom=scale_mom,
        cutoff=cutoff,
        force_as_grad=force_as_grad,
        model_version=model_version
    )
