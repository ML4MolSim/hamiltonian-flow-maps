import jax
import jax.numpy as jnp
import jax.random as jr
import logging

from functools import partial
from hfm import utils
from hfm.datasets.utils import jraph_force_temperature_fn, jraph_get_temperature, jraph_remove_global_rotation_3d, jraph_rotation_augmentation, jraph_stationary
from hfm.jraph_utils import get_batch_segments, get_node_padding_mask, get_number_of_graphs


class HFMDataset:
    def __init__(
            self, 
            force_dataset, 
            temperature_mean: float, 
            temperature_std: float = 0.0,
            n_dim: int = 3,
            rotation_augmentation: bool = True,
            zero_momenta_p: float = 0.0,
            load_momenta_from_force_dataset: bool = False,
            zero_rot_p: float = 0.0,
            zero_drift: bool = True,
        ):
        """
        Initializes the HFMDataset. It first samples a temperature value according to a normal distribution with
            `mean=temperature_mean` and `std=temperature_std` and then uses this temperature as target temperature 
            for sampling from the Maxwell Boltzmann distribution.

        Args:
            force_dataset: The underlying dataset.
            temperature_mean: The mean temperature for sampling in Kelvin.
            temperature_std: The standard deviation of the temperature for sampling in Kelvin. Defaults to zero, i.e., target temperature
                for the Maxwell Boltzmann distribution will be equal to temperature_mean.
            n_dim: The number of dimensions for the system (e.g., 2 for 2D, 3 for 3D).
            rotation_augmentation: Use data augmentation with random rotations.
            zero_momenta_p: Probability for setting all momenta to zero during training.
        """
    
        self.temperature_mean = temperature_mean
        self.temperature_std = temperature_std
        self.force_dataset = force_dataset
        self.n_dim = n_dim
        self.rotation_augmentation = rotation_augmentation
        self.zero_momenta_p = zero_momenta_p
        self.load_momenta_from_force_dataset = load_momenta_from_force_dataset
        self.zero_rot_p = zero_rot_p
        self.zero_drift = zero_drift

        if self.temperature_mean != 500.0 or self.temperature_std != 150.0:
            logging.warning(
                "Temperature mean or standard deviation has changed. "
                "You may need to update `velocities_embed_max_value` to twice the new 3-sigma interval."
            )

    @partial(jax.jit, static_argnames=['self'])
    def iterate(self, rng, graph):
        """
        Generates a sample for variable time molecular dynamics training. First, it samples
        a temperature according to a normal distribution with mean `temperature_mean` and standard deviation
        `temperature_std`. This temperature is then used as target temperature for sampling velocities 
        according to the Maxwell Boltzmann distribution. 

        Args:
            rng: JAX random key.
            graph: A jraph tuple containing batched data ('x' for positions and 'f' for forces).

        Returns:
            A jraph tuple containing 'x' (rotated positions), 'f' (rotated forces),
            'p' (momenta), 'v' (velocities), and 'T' (temperatures).
        """        
        num_graphs = get_number_of_graphs(graph)
        batch_segments = get_batch_segments(graph)
        rng_temp, rng_mom, rng_rot, rng_zero_mom, rng_zero_rot = jr.split(rng, 5)

        temp_K = jr.normal(
            rng_temp,
            shape=(num_graphs, 1)
        ) * self.temperature_std + self.temperature_mean
        temp_K = jnp.clip(temp_K, min=0.0)  # prevent negative temperatures
        temp_nodes = temp_K[batch_segments]  # (num_nodes, 1)

        # p is already in graph.nodes otherwise
        if not self.load_momenta_from_force_dataset:
            # this effectively means that we treat all atoms as individual samples
            # in the batch, but is ok since MB distribution is per-atom
            graph.nodes["p"] = utils.maxwell_boltzmann_distribution(
                rng_mom, 
                graph.nodes["masses"][:, None], 
                temp_nodes,
                n_dim=self.n_dim,
                force_temperature=False
            ).reshape(-1, self.n_dim)

        if self.zero_rot_p > 0.0:
            graph_zero_rot = jraph_remove_global_rotation_3d(graph, force_temperature=False) # we only change some graphs, dont enforce temp.
            zero_rot = jax.random.uniform(rng_zero_rot, shape=(num_graphs, 1))
            zero_rot_nodes = zero_rot[batch_segments]
            graph.nodes["p"] = jnp.where(zero_rot_nodes < self.zero_rot_p, 
                                        graph_zero_rot.nodes["p"], graph.nodes["p"])

        if self.zero_drift:
            # here we actually change the ndof by removing a substantial drift
            # not just a small correction
            temp0 = jraph_get_temperature(graph, n_dim=self.n_dim, zero_drift=False, zero_rot=False)
            graph = jraph_stationary(graph, force_temperature=False)
            graph = jraph_force_temperature_fn(graph, temp0, n_dim=self.n_dim, zero_drift=True, zero_rot=False)

        zero_mom = jax.random.uniform(rng_zero_mom, shape=(num_graphs, 1))
        zero_mom_nodes = zero_mom[batch_segments]
        graph.nodes["p"] = jnp.where(zero_mom_nodes < self.zero_momenta_p, 
                                     jnp.zeros_like(graph.nodes["p"]), graph.nodes["p"])

        if self.rotation_augmentation:
            graph = jraph_rotation_augmentation(rng_rot, graph, rotation_props=["x", "f", "p"])

        # avoid div by 0
        node_mask = get_node_padding_mask(graph)
        graph.nodes["masses"] = jnp.where(node_mask[:, None], graph.nodes["masses"], 1.)
        graph.nodes["v"] = graph.nodes["p"] / graph.nodes["masses"]

        # mask out padding for p, v
        graph.nodes["p"] = jnp.where(node_mask[:, None], graph.nodes["p"], 0.)
        graph.nodes["v"] = jnp.where(node_mask[:, None], graph.nodes["v"], 0.)

        return graph

    def next_epoch(self, rng, batch_size):
        rng, rng_shuffle = jax.random.split(rng, 2)
        for graph in self.force_dataset.next_epoch(rng_shuffle, batch_size):
            assert graph.nodes['x'].shape[-1] == self.n_dim, f"HFMDataset has been initialized with {self.n_dim=}. received positions with n_dim={graph.nodes['x'].shape[-1]}"
            rng, rng_iterate = jax.random.split(rng, 2)
            yield self.iterate(rng_iterate, graph)

    def get_example_batch(self, rng, batch_size):        
        rng_iterate, rng_example = jax.random.split(rng, 2)
        example_graph = self.iterate(rng_iterate, self.force_dataset.get_example_batch(rng_example, batch_size))
        assert example_graph.nodes['x'].shape[-1] == self.n_dim, f"HFMDataset has been initialized with {self.n_dim=}. received positions with n_dim={example_graph.nodes['x'].shape[-1]}"
        
        return example_graph

    def __len__(self):
        return len(self.force_dataset)
