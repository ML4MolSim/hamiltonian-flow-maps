import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
from hfm.jraph_utils import get_node_padding_mask

from hfm import utils
from hfm.backbones import embedding
from .utils import promote_to_e3x

import logging

from jaxtyping import Array
from typing import Optional, Sequence

from hfm.backbones import base

from .base import BaseReadout
from .base import GenerativeLayer
from .base import BaseNodeEmbedding
from .base import BaseEdgeEmbedding
from .base import BaseTimeEmbedding
from .base import FeatureRepresentations

from .utils import build_graph, get_activation_fn, get_cutoff_value
from .utils import MLP
from .utils import ZBLRepulsion
from .utils import modulate_adaLN
from ..jraph_utils import get_batch_segments, get_number_of_graphs, get_number_of_nodes
from packaging.version import Version


class SimpleReadout(nn.Module):
    activation_fn: str
    use_adaptive_layernorm: bool = True
    use_task_specific_adaln: bool = False
    zbl_repulsion_bool: bool = False
    cutoff: Optional[float] = None
    cutoff_fn: str = 'smooth_cutoff'
    model_version: str = "0.0.1"

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
        assert features_nodes.shape[1] == 1, 'Parity must be 1.'
        assert features_nodes.shape[2] == 1, 'Maximal degree must be 0.'
        features_nodes = jnp.squeeze(features_nodes, axis=(1, 2))  # (num_nodes, num_features)

        num_features = features_nodes.shape[-1]

        if self.use_adaptive_layernorm:
            c = jnp.squeeze(features_time, axis=(1, 2))  # (num_nodes, num_features)
            c = nn.LayerNorm()(c)  # (num_nodes, num_features)

            act_fn = get_activation_fn(self.activation_fn)

            if self.use_task_specific_adaln:
                shift1, scale1, shift2, scale2, shift3, scale3 = jnp.split(
                    nn.Dense(
                        features=6 * num_features,
                        kernel_init=jax.nn.initializers.zeros
                    )(
                        act_fn(c)
                    ),
                    indices_or_sections=6,
                    axis=-1
                )  # (num_nodes, num_features), (num_nodes, num_features)

                y_velocity = modulate_adaLN(
                    x=nn.LayerNorm(use_bias=False, use_scale=False)(x=features_nodes),
                    shift=shift1,
                    scale=scale1
                )

                y_force = modulate_adaLN(
                    x=nn.LayerNorm(use_bias=False, use_scale=False)(x=features_nodes),
                    shift=shift2,
                    scale=scale2
                )

                y_energy = modulate_adaLN(
                    x=nn.LayerNorm(use_bias=False, use_scale=False)(x=features_nodes),
                    shift=shift3,
                    scale=scale3
                )
            else:
                shift, scale = jnp.split(
                    nn.Dense(
                        features=2 * num_features,
                        kernel_init=jax.nn.initializers.zeros
                    )(
                        act_fn(c)
                    ),
                    indices_or_sections=2,
                    axis=-1
                )  # (num_nodes, num_features), (num_nodes, num_features)

                y = modulate_adaLN(
                    x=nn.LayerNorm(use_bias=False, use_scale=False)(x=features_nodes),
                    shift=shift,
                    scale=scale
                )
        else:
            y = nn.LayerNorm()(features_nodes)

        # mean_v, mean_f, energy = jnp.split(
        #     nn.Dense(features=7)(y),
        #     indices_or_sections=[3, 6],
        #     axis=-1
        # )
        
        mean_v = MLP(
            num_layers=2,
            num_features=(num_features, 3),
            activation_fn=self.activation_fn,
            use_bias=True
        )(y if not self.use_task_specific_adaln else y_velocity)

        mean_f = MLP(
            num_layers=2,
            num_features=(num_features, 3),
            activation_fn=self.activation_fn,
            use_bias=True
        )(y if not self.use_task_specific_adaln else y_force)

        energy = MLP(
            num_layers=2,
            num_features=(num_features, 1),
            activation_fn=self.activation_fn,
            use_bias=True
        )(y if not self.use_task_specific_adaln else y_energy)

        if self.zbl_repulsion_bool:
            energy_zbl = ZBLRepulsion(
                cutoff=self.cutoff,
                cutoff_fn=self.cutoff_fn,
            )(
                graph=graph
            ) # (num_nodes, 1)

            energy += energy_zbl # (num_nodes, 1)
        
        # # Mask out contribution from the padding nodes.
        node_mask = get_node_padding_mask(graph) # (num_nodes)
        energy = jnp.where(jnp.expand_dims(node_mask, axis=-1), energy, 0.0) # (num_nodes, 1)

        # sum energy contributions over atoms in each graph
        num_graphs = get_number_of_graphs(graph)
        batch_segments = get_batch_segments(graph)
        energy = jax.ops.segment_sum(energy, batch_segments, num_graphs)

        return mean_v, mean_f, energy
        

class DiTLayer(nn.Module):
    num_heads: int
    num_features_mlp: int
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
        features_nodes = features.nodes  # (num_nodes, 1, 1, num_features)
        features_edges = features.edges  # (num_nodes, 1, 1, num_features)

        num_features = features_nodes.shape[-1]

        assert features_nodes.ndim == 4, 'Features are assumed to be in the e3x convention.'
        assert features_nodes.shape[1] == 1, 'Parity must be 1.'
        assert features_nodes.shape[2] == 1, 'Maximal degree must be 0.'

        features_nodes = jnp.squeeze(features_nodes, axis=(1, 2))

        if features_edges is not None:
            assert features_edges.shape[1] == 1, 'Parity must be 1.'
            assert features_edges.shape[2] == 1, 'Maximal degree must be 0.'

            features_edges = jnp.squeeze(features_edges, axis=(1, 2))

        if self.use_adaptive_layernorm:
            # Calculate the shift and scale parameters for adaLN and adaLN-Zero
            c = jnp.squeeze(features_time, axis=(1, 2))  # (num_nodes, num_features)
            c = nn.LayerNorm()(c)  # (num_nodes, num_features)

            act_fn = get_activation_fn(self.activation_fn)

            gamma1, beta1, alpha1, gamma2, beta2, alpha2 = jnp.split(
                nn.Dense(
                    features=6 * num_features,
                    kernel_init=jax.nn.initializers.zeros
                )(
                    act_fn(c)
                ),
                indices_or_sections=6,
                axis=-1
            )  # 6 times (num_nodes, num_features)

            features_nodes_pre_attention = modulate_adaLN(
                x=nn.LayerNorm(use_bias=False, use_scale=False)(features_nodes),
                scale=gamma1,
                shift=beta1
            )
        else:
            features_nodes_pre_attention = nn.LayerNorm()(features_nodes)

        with jax.debug_nans(False):
            with jax.debug_infs(False):
                features_nodes_post_att = e3x.nn.SelfAttention(
                    num_heads=self.num_heads,
                    max_degree=0,
                    use_fused_tensor=False,
                    include_pseudotensors=False,
                    use_relative_positional_encoding_qk=self.relative_embedding_qk_bool,
                    use_relative_positional_encoding_v=self.relative_embedding_v_bool
                )(
                    promote_to_e3x(features_nodes_pre_attention),
                    promote_to_e3x(features_edges) if features_edges is not None else None,
                    cutoff_value=cutoff_value,
                    dst_idx=dst_idx,
                    src_idx=src_idx,
                    num_segments=num_nodes
                )  # (num_node, 1, 1, num_features)

                features_nodes_post_att = features_nodes_post_att.squeeze(
                    axis=(1, 2)
                )  # (num_node, num_features)

                # there might be cases where nodes don't receive any inputs due to only zero-cutoff edges
                # we have to mask them out to avoid NaNs
                features_nodes_post_att = jnp.where(jraph.segment_mean(cutoff_value, dst_idx, num_nodes)\
                                                    .reshape(-1,1) < 1e-5, features_nodes_pre_attention, features_nodes_post_att)
                
        if self.use_adaptive_layernorm:
            # Skip connection with scaling.
            features_nodes = features_nodes + features_nodes_post_att * alpha1  # (num_node, num_features)

            features_nodes_pre_mlp = modulate_adaLN(
                x=nn.LayerNorm(use_bias=False, use_scale=False)(features_nodes),
                scale=gamma2,
                shift=beta2
            )  # (num_node, num_features)
        else:
            # Skip connection
            features_nodes = features_nodes + features_nodes_post_att  # (num_node, num_features)

            features_nodes_pre_mlp = nn.LayerNorm()(
                features_nodes
            )  # (num_node, num_features)

        features_nodes_post_mlp = MLP(
            num_features=[self.num_features_mlp, num_features],
            num_layers=2,
            activation_fn=self.activation_fn_mlp,
        )(
            features_nodes_pre_mlp
        )  # (num_node, num_features)

        if self.use_adaptive_layernorm:
            # Skip connection with scaling.
            features_nodes = features_nodes + features_nodes_post_mlp * alpha2  # (num_node, num_features)
        else:
            # Skip connection
            features_nodes = features_nodes + features_nodes_post_mlp  # (num_node, num_features)

        return FeatureRepresentations(
            nodes=promote_to_e3x(features_nodes),
            edges=promote_to_e3x(features_edges) if features_edges is not None else None
        )
    

class DiTModel(nn.Module):
    node_embedding: BaseNodeEmbedding
    time_embedding: BaseTimeEmbedding
    layers: Sequence[GenerativeLayer]
    readout: BaseReadout
    scale_mom: float
    scale_pos: float
    edge_embedding: Optional[BaseEdgeEmbedding] = None
    name: str
    cutoff: Optional[float] = None
    cutoff_fn: str = 'smooth_cutoff'
    force_as_grad: bool = False
    model_version: str = "0.0.1"

    def _internal_fwd(
            self,
            time_latent, graph,
            deterministic: bool = False
    ):
        # org_shape = x.shape
        # batch_jraph = build_graph({"x": x * self.scale_pos, "p": p * self.scale_mom, "atomic_numbers": atomic_numbers, "masses": masses}, x.shape[0], x.shape[1])
        
        batch_segments = get_batch_segments(graph)
        node_mask = get_node_padding_mask(graph)
        #time_latent = t.reshape((x.shape[0],))
        time_latent = jnp.take(time_latent, batch_segments)
        time_latent = jnp.where(node_mask, time_latent, 0.)

        features_nodes = self.node_embedding(
            graph=graph
        )  # (num_atoms, *dims)

        features_time = self.time_embedding(
            time_latent=time_latent,
            graph=graph,
        )  # (num_atoms)

        if self.edge_embedding is not None:
            features_edges = self.edge_embedding(graph=graph)
        else:
            features_edges = None

        features = FeatureRepresentations(
            nodes=features_nodes, edges=features_edges
        )

        if self.cutoff is not None:
            cutoff_value = get_cutoff_value(graph=graph, cutoff_fn=self.cutoff_fn, cutoff=self.cutoff)
        else:
            cutoff_value = None

        for n in range(len(self.layers)):
            encoder = self.layers[n].encoder

            features = encoder(
                graph=graph,
                features=features,
                features_time=features_time,
                cutoff_value=cutoff_value,
            )

        mean_v_pred, mean_f_pred, energy = self.readout(
            graph=graph,
            features=features,
            features_time=features_time,
        )

        # Reshape the output to match the original input shape
        # mean_v_pred = mean_v_pred.reshape(org_shape)
        # mean_f_pred = mean_f_pred.reshape(org_shape)
        return mean_v_pred, mean_f_pred, energy
    
    def _internal_fwd_grad(
            self,
            time_latent, graph,
            deterministic: bool = False,
    ):
        def _fwd(pos):
            graph_comp = jax.tree.map(lambda x: x, graph)
            graph_comp.nodes["x"] = pos

            _, _, energy = self._internal_fwd(
                time_latent, graph_comp, deterministic=deterministic
            )
            return (
                -jnp.sum(energy),
                energy,
            )

        energy_and_forces = jax.value_and_grad(
            _fwd, has_aux=True
        )
        (_, energy), forces = energy_and_forces(graph.nodes["x"])

        # don't predict mean velocity for now..
        # replace with zeros
        return jnp.zeros_like(forces), forces, energy

    @nn.compact
    def __call__(
            self,
            time_latent, graph,
            deterministic: bool = False,
    ):
        # assert x.ndim == 3, "x and p must be (BS, num_nodes, num_dimensions)"
        # assert x.shape == p.shape, "x and p must have the same shape"

        if self.force_as_grad:
            # use model as force field
            return self._internal_fwd_grad(
                time_latent, graph,
                deterministic=deterministic
            )
        else:
            return self._internal_fwd(
                time_latent, graph,
                deterministic=deterministic
            )
    

def make_molecular_DiT(
        num_layers: int,
        num_heads: int,
        num_features_head: int,
        cutoff: float,
        cutoff_fn: str = 'smooth_cutoff',
        num_features_mlp: int = None,
        use_adaptive_layernorm: bool = True,
        rpe_radial_basis: str = 'basic_fourier',
        rpe_num_basis: int = 10,
        activation_fn_mlp: str = 'gelu',
        activation_fn: str = 'silu',
        scale_pos: float = 1.0,
        scale_mom: float = 1.0,
        name: str = "DiT",
        use_mass_cond: bool = False,
        use_velocity_cond: bool = False,
        use_task_specific_adaln: bool = False,
        embed_positions_absolute: bool = False,
        embed_velocities_absolute: bool = True,
        embed_velocities_relative: bool = False,
        embed_masses: bool = False,
        velocities_embed_encode_magnitude: bool = False,
        velocities_embed_num_basis: int = 8,
        velocities_embed_max_frequency: float = 4*jnp.pi,
        velocities_embed_max_value: float = 1.55,
        force_as_grad: bool = False,
        zbl_repulsion: bool = False,
        model_version: str = "0.0.1",
        **kwargs
):
    
    if "embed_positions_relative" in kwargs:
        raise DeprecationWarning(
            f'`embed_positions_relative` has been depracated, since this should be always done. At least for the moment.'
        )

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
    time_embedding = embedding.TimeEmbeddingV2(
        num_features=num_features,
        activation_fn=activation_fn,
        use_mass_cond=use_mass_cond,
        use_velocity_cond=use_velocity_cond,
        velocities_num_basis=velocities_embed_num_basis,
        velocities_max_value=velocities_embed_max_value,
    )

    # Node embedding
    node_embedding = embedding.DiTNodeEmbed(
        num_features=num_features,
        activation_fn=activation_fn,
        mass_embedding_bool=embed_masses,
        positional_embedding_bool=embed_positions_absolute,
        velocities_embedding_bool=embed_velocities_absolute,
        velocities_encode_magnitude_bool=velocities_embed_encode_magnitude,
        velocities_num_basis=velocities_embed_num_basis,
        velocities_max_frequency=velocities_embed_max_frequency
    )

    # Edge embedding
    edge_embedding = embedding.DiTEdgeEmbed(
        num_features=num_features,
        activation_fn=activation_fn,
        cutoff=cutoff,
        cutoff_fn=cutoff_fn,
        radial_basis=rpe_radial_basis,
        num_basis=rpe_num_basis,
        embed_rel_velocities_bool=embed_velocities_relative,
        model_version=model_version
    )

    # Readout block
    readout_block = SimpleReadout(
        activation_fn=activation_fn,
        use_adaptive_layernorm=use_adaptive_layernorm,
        use_task_specific_adaln=use_task_specific_adaln,
        zbl_repulsion_bool=zbl_repulsion,
        cutoff=cutoff,
        cutoff_fn=cutoff_fn,
        model_version=model_version
    )

    layers = []

    for _ in range(num_layers):
        layers.append(
            base.GenerativeLayer(
                encoder=DiTLayer(
                    num_heads=num_heads,
                    num_features_mlp=num_features_mlp,
                    activation_fn_mlp=activation_fn_mlp,
                    activation_fn=activation_fn,
                    use_adaptive_layernorm=use_adaptive_layernorm,
                    relative_embedding_qk_bool=True,
                    relative_embedding_v_bool=True,
                    model_version = model_version
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
        cutoff_fn=cutoff_fn,
        force_as_grad=force_as_grad,
        model_version=model_version
    )
