import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from functools import partial
from jax import lax
from jax.nn.initializers import constant
from jax.nn import softplus
from typing import Callable, Sequence, Union
from typing import Sequence, Union

from hfm import jraph_utils, utils


def safe_mask(mask, fn: Callable, operand: jnp.ndarray, placeholder: float = 0.) -> jnp.ndarray:
    """
    Safe mask which ensures that gradients flow nicely. See also
    https://github.com/google/jax-md/blob/b4bce7ab9b37b6b9b2d0a5f02c143aeeb4e2a560/jax_md/util.py#L67

    Args:
        mask (array_like): Array of booleans.
        fn (Callable): The function to apply at entries where mask=True.
        operand (array_like): The values to apply fn to.
        placeholder (int): The values to fill in if mask=False.

    Returns: New array with values either being the output of fn or the placeholder value.

    """
    masked = jnp.where(mask, operand, 0)
    return jnp.where(mask, fn(masked), placeholder)


def soft_gaussian(
    x,
    num: int,
    limit: float = 1.0,
    cutoff: bool = True,
):
    if not cutoff:
        values = jnp.linspace(0.0, limit, num)
        step = values[1] - values[0]
    else:
        values = jnp.linspace(0.0, limit, num + 2)
        step = values[1] - values[0]
        values = values[1:-1]

    diff = (x[..., None] - values) / step

    return (num**0.5) * jnp.exp(-diff**2) / 1.12 


def get_activation_fn(name: str):
    if name == 'identity':
        return lambda x: x
    else:
        return getattr(
            jax.nn,
            name
        )


def get_e3x_activation_fn(name: str):
    if name == 'identity':
        return lambda x: x
    else:
        return getattr(
            e3x.nn,
            name
        )


def get_e3x_radial_basis_fn(
        name: str, 
        cutoff: float,
        gamma=1.0, 
        cuspless=True, 
        use_exponential_weighting=True,
        kind='shifted', 
        use_reciprocal_weighting=True,
):
    radial_basis_fn = getattr(e3x.nn, name)
    
    # get prefix before the underscore
    if name == 'sinc':
        # 'sinc' is a special case without an underscore
        prefix = name 
    else:
        prefix = name.split("_", 1)[0]

    if prefix in ("basic", "sinc"):
        if cutoff is None:
            raise ValueError(f"{name} requires a finite cutoff as `limit`.")
        return partial(radial_basis_fn, limit=cutoff)
    
    elif prefix == "exponential":
        return partial(
            radial_basis_fn, 
            gamma=gamma, 
            cuspless=cuspless, 
            use_exponential_weighting=use_exponential_weighting,
        )
    
    elif prefix == "reciprocal":
        return partial(
            radial_basis_fn, 
            kind=kind, 
            use_reciprocal_weighting=use_reciprocal_weighting,
        )
    
    else:
        raise ValueError("Unknown basis function {name}")


def get_cutoff_value(graph, cutoff_fn: str, cutoff: float):
    src_idx = graph.senders # (num_pairs)
    dst_idx = graph.receivers # (num_pairs)
    positions = graph.nodes['x'] # (num_nodes, 3)

    # Calculate the displacements.
    distances = e3x.ops.norm(positions[src_idx] - positions[dst_idx], axis=-1) # (num_pairs)
    
    # Calculate the cutoff mask.
    cutoff_fn = getattr(e3x.nn, cutoff_fn)
    cutoff_value = cutoff_fn(distances, cutoff=cutoff) # (num_pairs)

    return cutoff_value


def modulate_adaLN(x, shift, scale):
    """

    Args:
        x (): Features to be modulated. (num_atoms, num_features)
        shift (): Shift to be added. (num_atoms, num_features)
        scale (): Scale to be multiplied. (num_atoms, num_features)

    Returns:

    """

    if not x.shape == scale.shape == shift.shape:
        raise ValueError(
            f'Shape of features, scale and shift must be identical. '
            f'Received {x.shape=}, {scale.shape=} and {shift.shape=}.'
        )

    return x * (1 + scale) + shift


def modulate_E3adaLN(x, shift, scale):
    """

    Args:
        x (): Features to be modulated. (num_atoms, parity, (max_degree + 1)**2, num_features)
        shift (): Shift to be added. (num_atoms, 1, 1, num_features)
        scale (): Scale to be multiplied. (num_atoms, num_features)

    Returns:

    """

    x_scaled = broadcast_equivariant_multiplication(
        factor=1 + scale,
        tensor=x
    )

    return e3x.nn.add(
        x_scaled,
        shift
    )


class MLP(nn.Module):
    num_layers: int = 2
    activation_fn: str = 'identity'
    num_features: Union[int, Sequence[int]] = None
    use_bias: bool = True
    output_is_zero_at_init: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = False):

        activation_fn = get_activation_fn(
            name=self.activation_fn
        )

        if type(self.num_features) == list or type(self.num_features) == tuple:
            num_features = self.num_features
        else:
            num_features = x.shape[-1] if self.num_features is None else self.num_features
            num_features = [num_features] * self.num_layers

        for n in range(self.num_layers):

            if self.output_is_zero_at_init is True and n == self.num_layers - 1:
                # Initialize the last layer with zeros such that the output of the MLP
                kernel_init = jax.nn.initializers.zeros
            else:
                kernel_init = jax.nn.initializers.lecun_normal()

            x = nn.Dense(
                num_features[n],
                use_bias=self.use_bias,
                kernel_init=kernel_init
            )(
                x
            )

            # do not apply activation / dropout in the last layer.
            if n < self.num_layers - 1:
                x = activation_fn(
                    x
                )

                if self.dropout_rate > 0.0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        return x


class E3MLP(nn.Module):
    num_layers: int = 2
    activation_fn: str = 'identity'
    num_features: Union[int, Sequence[int]] = None
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):

        if type(self.num_features) == list or type(self.num_features) == tuple:
            num_features = self.num_features
        else:
            num_features = x.shape[-1] if self.num_features is None else self.num_features
            num_features = [num_features] * self.num_layers

        activation_fn = get_e3x_activation_fn(
            name=self.activation_fn
        )

        for n in range(self.num_layers):
            x = e3x.nn.Dense(
                num_features[n],
                use_bias=self.use_bias
            )(
                x
            )

            # do not apply activation in the last layer.
            if n < self.num_layers - 1:
                x = activation_fn(
                    x
                )

        return x


def promote_to_e3x(x: jnp.ndarray) -> jnp.ndarray:
    """
  Promote an invariant node representation to a tensor that matches the shape
  convention of e3x, i.e. adding an axis for parity and irreps.

    Args:
      x: Tensor of shape (n, F)

    Returns: Tensor of shape (n, 1, 1, F)
  """
    assert x.ndim == 2
    return x[:, None, None, :]

def make_to_jraph_fn(num_node):
    # pre-compute edges
    #num_node = sample[list(sample.keys())[0]].shape[0]
    node_range = jnp.arange(num_node)
    a = node_range.repeat(num_node)
    b = jnp.tile(node_range, num_node)
    edges = jnp.stack([a, b], axis=0)
    # mask = ~(edges[0] == edges[1])
    # edges = edges[:, mask]

    # Compute the difference and get the sorted indices
    # This is the jit-able way to remove self-interactions
    diff = edges[0] - edges[1]
    sorted_idx = jnp.argsort(jnp.abs(diff))
    edges = edges[:, sorted_idx][:, num_node:]

    def to_jraph(nodes, globals): 
        return jraph.GraphsTuple(
            n_node=jnp.array([num_node]),
            n_edge=jnp.array([num_node * (num_node - 1)]),
            receivers=edges[0],
            senders=edges[1],
            nodes=nodes,
            globals=globals,
            edges={},
        )
    
    return to_jraph

def make_to_jraph_batch_fn(num_graph, num_node, global_properties):
    # assuming that all features are node features
    # and data is already batched (w static shapes)
    to_jraph = make_to_jraph_fn(num_node)

    def _extract_sample(idx, b, gps):
        nodes = {k: v[idx] for k, v in b.items() if k not in gps}
        globals = {k: v[idx:idx+1] for k, v in b.items() if k in gps} # keep batch dim
        return nodes, globals

    def to_jraph_batch(batch: dict):
        graphs = [to_jraph(*_extract_sample(idx, batch, global_properties)) for idx in range(num_graph)]
        batch_graph = jraph.batch(graphs)
        
        # Add a padding graph for compatibility with dynamically padded batches
        tree_nodes_pad = (
            lambda leaf: jnp.zeros((1,) + leaf.shape[1:], dtype=leaf.dtype))
        tree_edges_pad = (
            lambda leaf: jnp.zeros((0,) + leaf.shape[1:], dtype=leaf.dtype))
        tree_globs_pad = (
            lambda leaf: jnp.zeros((1,) + leaf.shape[1:], dtype=leaf.dtype))

        padding_graph = jraph.GraphsTuple(
            n_node=jnp.array([1], dtype=jnp.int32),
            n_edge=jnp.array([0], dtype=jnp.int32),
            nodes=jax.tree.map(tree_nodes_pad, batch_graph.nodes),
            edges=jax.tree.map(tree_edges_pad, batch_graph.edges),
            globals=jax.tree.map(tree_globs_pad, batch_graph.globals),
            senders=jnp.zeros(0, dtype=jnp.int32),
            receivers=jnp.zeros(0, dtype=jnp.int32),
        )

        return jraph.batch([batch_graph, padding_graph])

    return to_jraph_batch

@partial(jax.jit, static_argnums=(1,2,3))
def build_graph(batch, num_graph: int, num_node: int, global_properties=()):
    fn = make_to_jraph_batch_fn(num_graph, num_node, global_properties)
    return fn(batch)

def get_max_degree_from_tensor_e3x(x):
    return int(np.rint(np.sqrt(x.shape[-2]) - 1).item())


def broadcast_equivariant_multiplication(factor, tensor):
    max_degree_tensor = get_max_degree_from_tensor_e3x(tensor)
    max_degree_factor = factor.shape[-2] - 1

    assert factor.shape[-1] == tensor.shape[-1], \
        f'Feature dimensions must align. Received {factor.shape=} and {tensor.shape=}'

    assert len(tensor) == len(factor), \
        f'Leading axis must align. Received {len(factor)=} and {len(tensor)=}'

    assert max_degree_factor == max_degree_tensor, \
        f'Max degree must align. Received {max_degree_factor=} and {max_degree_tensor=}'

    repeats_factor = [2 * ell + 1 for ell in range(max_degree_factor + 1)]

    return jnp.repeat(
        factor,
        axis=-2,
        repeats=np.array(repeats_factor),
        total_repeat_length=(max_degree_tensor + 1) * (max_degree_tensor + 1)
    ) * tensor

def equivariant_concatenation(x, y):
    assert x.ndim == 4
    assert y.ndim == 4

    x_shape = x.shape
    y_shape = y.shape

    x_invariant_bool = x_shape[1:3] == (1, 1)
    y_invariant_bool = y_shape[1:3] == (1, 1)

    # Easy case of all dimensions equal.
    if x_shape[:3] == y_shape[:3]:
        return jnp.concatenate([x, y], axis=-1)

    # One of the the two is invariant.
    elif x_invariant_bool is True or y_invariant_bool is True:
        if x_invariant_bool:
            x = e3x.nn.add(x, jnp.zeros((*y.shape[:3], x.shape[-1])))  # hack to bring to correct shape

            # does the same as this code:
            # _x = jnp.zeros_like(y)
            # x = _x.at[:, 0, 0, :].set(jnp.squeeze(x, axis=(1, 2)))

            return jnp.concatenate([x, y], axis=-1)
        else:
            y = e3x.nn.add(y, jnp.zeros((*x.shape[:3], y.shape[-1])))  # hack to bring to correct shape

            # does the same as this code:
            # _y = jnp.zeros_like(x)
            # y = _y.at[:, 0, 0, :].set(jnp.squeeze(y, axis=(1, 2)))

            return jnp.concatenate([x, y], axis=-1)
    else:
        raise NotImplementedError(
            f'At the moment, equivariant concatenation is only supported for both features having same '
            f'max_degree and parity or one of the features invariant (P=1, max_degree=0) and the other arbitrary. '
            f'received {x.shape} and {y.shape}.'
        )
    

def make_degree_repeat_fn(degrees: Sequence[int], axis: int = -1):
    repeats = np.array([2 * y + 1 for y in degrees])
    repeat_fn = partial(np.repeat, repeats=repeats, axis=axis)
    return repeat_fn


class EquivariantLayerNorm(nn.Module):
    use_scale: bool = True
    use_bias: bool = True

    bias_init: Callable = nn.initializers.zeros
    scale_init: Callable = nn.initializers.ones

    epsilon: float = 1e-6
    param_dtype = jnp.float32

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        """
            x.shape: (N, 1 or 2, (max_degree + 1)^2, features)
        """
        assert x.ndim == 4

        max_degree = int(np.rint(np.sqrt(x.shape[-2]))) - 1
        num_features = x.shape[-1]
        num_atoms = x.shape[-4]

        has_pseudotensors = x.shape[-3] == 2
        has_ylms = x.shape[-2] > 1

        if has_pseudotensors or has_ylms:
            plm_axes = x.shape[-3:-1]

            y = x.reshape(num_atoms, -1, num_features)  # (N, plm, features)
            y00, ylm = jnp.split(
                y,
                axis=1,
                indices_or_sections=np.array([1])
            )  # (N, 1, features), (N, plm - 1, features)

            # Construct the segment sum indices for summing over degree and parity channels.
            repeat_fn_even = make_degree_repeat_fn(degrees=list(range(1, max_degree + 1)))
            sum_idx_even = repeat_fn_even(np.arange(max_degree))

            if has_pseudotensors:
                repeat_fn_odd = make_degree_repeat_fn(degrees=list(range(max_degree + 1)))
                sum_idx_odd = repeat_fn_odd(np.arange(max_degree, 2 * max_degree + 1))
            else:
                sum_idx_odd = np.array([], dtype=sum_idx_even.dtype)

            sum_idx = np.concatenate([sum_idx_even, sum_idx_odd], axis=0)

            ylm_sum_squared = jax.vmap(
                partial(
                    jax.ops.segment_sum,
                    segment_ids=sum_idx,
                    num_segments=2 * max_degree + 1 if has_pseudotensors else max_degree
                )
            )(
                lax.square(ylm),
            )  # (N, parity * max_degree + 1 or max_degree, features)

            ylm_inv = safe_mask(
                ylm_sum_squared > self.epsilon,
                lax.sqrt,
                ylm_sum_squared
            )

            _, var_lm = nn.normalization._compute_stats(
                ylm_inv,
                axes=-1,
                dtype=None
            )  # (N, parity * max_degree + 1 or max_degree)

            mul_lm = lax.rsqrt(var_lm + jnp.asarray(self.epsilon, dtype=var_lm.dtype))
            # (N, parity * max_degree + 1 or max_degree)

            if self.use_scale:
                scales_lm = self.param(
                    'scales_lm',
                    self.scale_init,
                    (var_lm.shape[-1], ),
                    self.param_dtype
                )  # (parity * max_degree + 1 or max_degree)

                mul_lm = mul_lm * scales_lm  # (N, parity * max_degree + 1 or max_degree)

            mul_lm = jnp.expand_dims(mul_lm, axis=-1)  # (N, parity * max_degree + 1 or max_degree, 1)

            ylm = ylm * mul_lm[:, sum_idx, :]  # (N, plm - 1, features)

            y00 = nn.LayerNorm(
                use_scale=self.use_scale,
                use_bias=self.use_bias,
                scale_init=self.scale_init,
                bias_init=self.bias_init
            )(y00)  # (N, 1, features)

            y = jnp.concatenate([y00, ylm], axis=1)  # (N, plm, features)
            return y.reshape(num_atoms, *plm_axes, num_features)  # (N, 1 or 2, (max_degree + 1)^2, features)
        else:
            return nn.LayerNorm(
                use_scale=self.use_scale,
                use_bias=self.use_bias,
                scale_init=self.scale_init,
                bias_init=self.bias_init
            )(x)  # (N, 1, features)


# Utility functions for ZBL Repulsion.

def softplus_inverse(x):
    return x + jnp.log(-jnp.expm1(-x))


def sigma(x):
    return safe_mask(x > 0, fn=lambda u: jnp.exp(-1. / u), operand=x, placeholder=0)


def switching_fn(x, x_on, x_off):
    c = (x - x_on) / (x_off - x_on)
    return sigma(1 - c) / (sigma(1 - c) + sigma(c))


class ZBLRepulsion(nn.Module):
    """
    Ziegler-Biersack-Littmark repulsion. Adapted from 
    https://github.com/thorben-frank/mlff/blob/v1.0/mlff/nn/observable/observable_sparse.py

    """
    cutoff: float
    cutoff_fn: str = 'smooth_cutoff'
    
    a0: float = 0.5291772105638411
    ke: float = 14.399645351950548

    @nn.compact
    def __call__(
        self,
        graph: jraph.GraphsTuple,
        *args,
        **kwargs
    ):
        """
        ZBL Repulsion.
        
        Args:
            positions (): atomic positions, (num_atoms, 3)
            atomic_numbers (): atomic numbers, (num_atoms, 1)

        Returns:
            Repulsion energy, (num_nodes, 1)
        """
        
        # Initial parameters. Can be adjusted during training.
        a1 = softplus(self.param('a1', constant(softplus_inverse(3.20000)), (1,)))  # shape: (1)
        a2 = softplus(self.param('a2', constant(softplus_inverse(0.94230)), (1,)))  # shape: (1)
        a3 = softplus(self.param('a3', constant(softplus_inverse(0.40280)), (1,)))  # shape: (1)
        a4 = softplus(self.param('a4', constant(softplus_inverse(0.20160)), (1,)))  # shape: (1)
        c1 = softplus(self.param('c1', constant(softplus_inverse(0.18180)), (1,)))  # shape: (1)
        c2 = softplus(self.param('c2', constant(softplus_inverse(0.50990)), (1,)))  # shape: (1)
        c3 = softplus(self.param('c3', constant(softplus_inverse(0.28020)), (1,)))  # shape: (1)
        c4 = softplus(self.param('c4', constant(softplus_inverse(0.02817)), (1,)))  # shape: (1)
        p = softplus(self.param('p', constant(softplus_inverse(0.23)), (1,)))  # shape: (1)
        d = softplus(self.param('d', constant(softplus_inverse(1 / (0.8854 * self.a0))), (1,)))  # shape: (1)

        c_sum = c1 + c2 + c3 + c4
        c1 = c1 / c_sum
        c2 = c2 / c_sum
        c3 = c3 / c_sum
        c4 = c4 / c_sum

        # Retrieve information.
        atomic_numbers = graph.nodes['atomic_numbers'] # (num_nodes, 1)
        atomic_numbers = jnp.squeeze(atomic_numbers, axis=-1) # (num_nodes)
        src_idx = graph.senders # (num_pairs)
        dst_idx = graph.receivers # (num_pairs)
        positions = graph.nodes['x'] # (num_nodes, 3)

        # Calculate the displacements.
        distances = e3x.ops.norm(positions[src_idx] - positions[dst_idx], axis=-1)  # (num_pairs)
        
        # Calculate cutoff function values.
        with jax.ensure_compile_time_eval():
            cutoff_fn = partial(getattr(e3x.nn, self.cutoff_fn), cutoff=self.cutoff)
        cut = cutoff_fn(distances) # (num_pairs)

        # Senders and receivers.
        src_idx = graph.senders # (num_pairs)
        dst_idx = graph.receivers # (num_pairs)
        
        # Atomic numbers to edges.
        z_dst = atomic_numbers[dst_idx] # (num_pairs)
        z_src = atomic_numbers[src_idx] # (num_pairs)
        
        # Core calculations.
        zd = z_dst * z_src / jnp.where(distances > 1e-5, d, 1) # (num_pairs)
        x = self.ke * cut * zd # (num_pairs)
        rzd = distances * (jnp.power(z_dst, p) + jnp.power(z_src, p)) * d # (num_pairs)
        y = c1 * jnp.exp(-a1 * rzd) + c2 * jnp.exp(-a2 * rzd) + c3 * jnp.exp(-a3 * rzd) + c4 * jnp.exp(-a4 * rzd) # (num_pairs)
        w = switching_fn(distances, x_on=0, x_off=1.5) # (num_pairs)

        # Bring all together and account for double counting.
        e_rep_edge = w * x * y / jnp.asarray(2, dtype=distances.dtype) # (num_pairs)
        
        # Aggregate edges into nodes.
        e_rep_atom = jax.ops.segment_sum(e_rep_edge, segment_ids=dst_idx, num_segments=len(atomic_numbers)) # (num_nodes)

        return jnp.expand_dims(e_rep_atom, axis=-1) # (num_nodes, 1)


def static_unbatch_jraph(batch_size, node_array):
    # Note: this works only for replicas of the same mol
    # Note: batch_size is num_graphs - 1 due to padding
    node_array = node_array[:-1]  # remove padding node
    return node_array.reshape(batch_size, -1, node_array.shape[-1])


def static_batch_jraph(node_array):
    # Note: this works only for replicas of the same mol
    node_array = node_array.reshape(-1, node_array.shape[-1])
    padding_node = jnp.zeros((1, node_array.shape[-1]))
    return jnp.concatenate([node_array, padding_node], axis=0)


def simulation_wrapper(model, params, t, x, p, atomic_numbers, masses, deterministic: bool = False, scale_pos=1.0, scale_mom=1.0):
    utils.verify_shapes(masses=masses, atomic_numbers=atomic_numbers, positions=x, momenta=p, time=t)

    batch_size = x.shape[0]
    batch_jraph = build_graph({"x": x * scale_pos, "p": p * scale_mom, "atomic_numbers": atomic_numbers, "masses": masses}, x.shape[0], x.shape[1])

    # mask out padding for p, v, masses
    node_mask = jraph_utils.get_node_padding_mask(batch_jraph)
    batch_jraph.nodes["masses"] = jnp.where(node_mask[:, None], batch_jraph.nodes["masses"], 1.)
    batch_jraph.nodes["p"] = jnp.where(node_mask[:, None], batch_jraph.nodes["p"], 0.)
    batch_jraph.nodes["x"] = jnp.where(node_mask[:, None], batch_jraph.nodes["x"], 0.)

    mean_v_pred, mean_f_pred, energy = model.apply(params, t, batch_jraph, deterministic=deterministic)
    
    # Reshape the output to match the original input shape
    mean_v_pred = static_unbatch_jraph(batch_size, mean_v_pred)
    mean_f_pred = static_unbatch_jraph(batch_size, mean_f_pred)
    return mean_v_pred, mean_f_pred, energy[:batch_size]
