import ase.units as units
import e3x
import jax
import jax.numpy as jnp
import jraph

from jaxtyping import PRNGKeyArray

from hfm import utils
from hfm.jraph_utils import get_batch_segments, get_graph_padding_mask, get_node_padding_mask
from hfm.jraph_utils import get_number_of_graphs

from omegaconf import OmegaConf


unit_map = {
    "eV": units.eV,
    "kB": units.kB,
    "Bohr": units.Bohr,
    "Hartree": units.Hartree,
    "Ang": units.Ang,
    "nm": units.nm,
    "kJ/mol": units.kJ / units.mol,
    "kcal/mol": units.kcal / units.mol,
    "fs": units.fs,
    "ps": units.fs * 1_000,
    "ns": units.fs * 1_000_000,
    "kcal/mol/Ang": units.kcal / units.mol / units.Ang,
    "kJ/mol/Ang": units.kJ / units.mol / units.Ang,
    "kJ/mol/nm": units.kJ / units.mol / units.nm,
    "eV/Ang": units.eV / units.Ang,
}


def parse_unit(str_or_number):
    if not isinstance(str_or_number, str):
        return str_or_number
    if str_or_number in unit_map:
        return unit_map[str_or_number]
    else:
        raise ValueError(f"Unit {str_or_number} not recognized. Available units: {list(unit_map.keys())}")


def ensure_resolver(name, func):
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, func)


def register_omegaconf_unit_resolvers():
    # register resolvers for ase units
    for unit, value in unit_map.items():
        ensure_resolver(unit.replace("/", "-"), lambda x, v=value: float(x) * v)
        ensure_resolver("print_" + unit.replace("/", "-"), lambda x, v=value, u=unit: f"{float(x) / v:.2f} {u}")

    def mul(a, b):
        return float(a) * float(b)
    ensure_resolver("mul", mul)


def rotate_graph(graph: jraph.GraphsTuple, rotation: jnp.ndarray, rotation_props: list):
    result = jax.tree.map(lambda x: x, graph)

    for p in rotation_props:
        # Rotate properties, if present.
        pval = result.nodes.get(p)
        pval_present = pval is not None
        if pval_present:
            pval_rot = jnp.einsum('bj,bjk->bk', pval, rotation)
            result.nodes[p] = pval_rot

    return result


def jraph_rotation_augmentation(
    rng: PRNGKeyArray,
    graph: jraph.GraphsTuple,
    rotation_props: list,
    return_rotation: bool = False
):
    n_dim = graph.nodes["x"].shape[-1] 

    num_graphs = get_number_of_graphs(graph)
    batch_segments = get_batch_segments(graph)

    if n_dim == 3:
        rot = e3x.so3.random_rotation(key=rng, num=num_graphs)  # (num_graphs, 3, 3) if num_graphs > 1 else (3, 3)
    elif n_dim == 2:
        rot = utils.random_rotation_2d(key=rng, num=num_graphs)  # (num_graphs, 2, 2) if num_graphs > 1 else (2, 2)
    elif n_dim == 1:
        rot = jnp.ones((num_graphs, 1, 1))
    else:
        raise ValueError(f"Number of dimensions must be 1, 2 or 3. received {n_dim=}.")

    if num_graphs == 1:
        rot = rot[None]  # (1, 3, 3)

    rot = rot[batch_segments]  # (num_nodes, 3, 3)
    graph = rotate_graph(graph, rot, rotation_props)

    if return_rotation:
        return graph, rot
    else:
        return graph
    

def jraph_center_pos(graph: jraph.GraphsTuple):
    batch_segments = get_batch_segments(graph)  # (num_nodes)
    num_graphs = get_number_of_graphs(graph)
    x = graph.nodes["x"]  # (num_nodes)
    x_mean_per_graph = jraph.segment_mean(x, batch_segments, num_graphs)

    graph_result = jax.tree.map(lambda x: x, graph)
    graph_result.nodes["x"] = x - x_mean_per_graph[batch_segments]
    return graph_result


def jraph_kinetic_energy(graph: jraph.GraphsTuple):
    momenta = graph.nodes['p']  
    masses = graph.nodes['masses']  
    num_graphs = get_number_of_graphs(graph)
    batch_segments = get_batch_segments(graph)
    node_mask = get_node_padding_mask(graph)

    # avoid div by zero
    masses = jnp.where(node_mask[:, None], masses, 1.)
    return 0.5 * jraph.segment_sum(jnp.sum((momenta**2) / masses, axis=-1, keepdims=True), batch_segments, num_graphs)


def jraph_get_dof(graph, n_dim=3, zero_drift: bool = False, zero_rot: bool = False):
    ndof = graph.n_node * n_dim

    # For single-atom graphs, keep full DOFs (no drift or rotation removal)
    single_atom = (graph.n_node == 1)

    # Apply drift and rotation corrections only for multi-atom systems
    return jnp.where(
        single_atom,
        ndof,
        ndof - (n_dim if zero_drift else 0) - (n_dim if (zero_rot and n_dim > 1) else 0)
    )[:, None]


def jraph_get_temperature(graph, unit="K", n_dim=3, zero_drift: bool = False, zero_rot: bool = False):
    ekin = jraph_kinetic_energy(graph)
    ndof = jraph_get_dof(graph, n_dim=n_dim, zero_drift=zero_drift, zero_rot=zero_rot)
    temp = (2 * ekin / (ndof * utils.kB))
    return utils.convert_temperature(temp, unit_from="K", unit_to=unit)


def jraph_force_temperature_fn(graph, temperature, unit="K", n_dim=3, zero_drift: bool = False, zero_rot: bool = False):
    batch_segments = get_batch_segments(graph)
    graph_mask = get_graph_padding_mask(graph)

    current_temp = jraph_get_temperature(graph, unit="eV", n_dim=n_dim, zero_drift=zero_drift, zero_rot=zero_rot)  # (B, 1)
    target_temp = utils.convert_temperature(temperature, unit_from=unit, unit_to="eV")
    
    # avoid div by zero
    current_temp = jnp.where(graph_mask[:, None], current_temp, 1.)
    current_temp = jnp.where(target_temp > utils.eps_temp, current_temp, 1.)
    
    # here we mask out again the cases where target_temp <= utils.eps_temp
    scale = jnp.where(target_temp > utils.eps_temp, target_temp / current_temp, 0.0)
    scale = jnp.sqrt(scale)[batch_segments]

    graph_result = jax.tree.map(lambda x: x, graph)
    graph_result.nodes["p"] = graph_result.nodes["p"] * scale
    return graph_result


def jraph_stationary(
        graph, 
        force_temperature=True,
        zero_drift: bool = False,
        zero_rot: bool = False):
    
    num_graphs = get_number_of_graphs(graph)
    batch_segments = get_batch_segments(graph)
    graph_mask = get_graph_padding_mask(graph)

    momenta = graph.nodes['p']  
    masses = graph.nodes['masses']  
    n_dim = momenta.shape[-1]

    total_p = jraph.segment_sum(momenta, batch_segments, num_graphs)
    total_m = jraph.segment_sum(masses, batch_segments, num_graphs)

    # avoid div by 0 for padding
    total_m = jnp.where(graph_mask[:, None], total_m, 1.)
    v0 = jnp.where(graph_mask[:, None], total_p / total_m, 0.)

    graph_result = jax.tree.map(lambda x: x, graph)
    graph_result.nodes['p'] = momenta - v0[batch_segments] * masses

    if force_temperature:
        # Compute per-graph current temperature (in eV)
        temp0 = jraph_get_temperature(graph, n_dim=n_dim, zero_drift=zero_drift, zero_rot=zero_rot)
        graph_result = jraph_force_temperature_fn(graph_result, temp0, n_dim=n_dim, zero_drift=zero_drift, zero_rot=zero_rot)

    return graph_result


def jraph_total_angular_momentum_3d(graph):    
    num_graphs = get_number_of_graphs(graph)
    batch_segments = get_batch_segments(graph)
    node_mask = get_node_padding_mask(graph)

    momenta = graph.nodes['p']
    masses = graph.nodes['masses']
    positions = graph.nodes['x']

    # avoid div by 0
    masses = jnp.where(node_mask[:, None], masses, 1)
    velocities = momenta / masses

    assert graph.nodes['x'].shape[-1] == 3, "jraph_total_angular_momentum_3d only works for 3D graphs."

    # --- Center of mass per graph ---
    total_mass = jraph.segment_sum(masses, batch_segments, num_segments=num_graphs)  # (B,)
    com = jraph.segment_sum(positions * masses, batch_segments, num_segments=num_graphs) / total_mass  # (B, 3)
    r = positions - com[batch_segments]  # (total_nodes, 3)

    # --- Angular momentum per graph ---
    return jraph.segment_sum(jnp.cross(r, masses * velocities), batch_segments, num_segments=num_graphs)  # (B, 3)


def jraph_remove_global_rotation_3d(graph: jraph.GraphsTuple,
                                    force_temperature=True,
                                    zero_drift: bool = False,
                                    zero_rot: bool = False):
    
    num_graphs = get_number_of_graphs(graph)
    batch_segments = get_batch_segments(graph)
    node_mask = get_node_padding_mask(graph)

    momenta = graph.nodes['p']
    masses = graph.nodes['masses']
    positions = graph.nodes['x']

    # avoid div by 0
    masses = jnp.where(node_mask[:, None], masses, 1)
    velocities = momenta / masses

    assert graph.nodes['x'].shape[-1] == 3, "jraph_remove_global_rotation_3d only works for 3D graphs."

    # --- Center of mass per graph ---
    total_mass = jraph.segment_sum(masses, batch_segments, num_segments=num_graphs)  # (B,)
    com = jraph.segment_sum(positions * masses, batch_segments, num_segments=num_graphs) / total_mass  # (B, 3)
    r = positions - com[batch_segments]  # (total_nodes, 3)

    # --- Angular momentum per graph ---
    L = jraph.segment_sum(jnp.cross(r, masses * velocities), batch_segments, num_segments=num_graphs)  # (B, 3)

    # --- Inertia tensor per graph ---
    masses_it = masses.squeeze(-1)
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    I11 = jraph.segment_sum(masses_it * (y**2 + z**2), batch_segments, num_graphs)
    I22 = jraph.segment_sum(masses_it * (x**2 + z**2), batch_segments, num_graphs)
    I33 = jraph.segment_sum(masses_it * (x**2 + y**2), batch_segments, num_graphs)
    I12 = -jraph.segment_sum(masses_it * x * y, batch_segments, num_graphs)
    I13 = -jraph.segment_sum(masses_it * x * z, batch_segments, num_graphs)
    I23 = -jraph.segment_sum(masses_it * y * z, batch_segments, num_graphs)

    I = jnp.stack([
        jnp.stack([I11, I12, I13], axis=-1),
        jnp.stack([I12, I22, I23], axis=-1),
        jnp.stack([I13, I23, I33], axis=-1)
    ], axis=-2)  # (B, 3, 3)

    # --- Solve for angular velocity per graph ---
    def solve_omega(Ii, Li):
        return jnp.linalg.solve(Ii + jnp.eye(3) * 1e-8, Li)

    omega = jax.vmap(solve_omega)(I, L)  # (B, 3)

    # --- Correct momenta ---
    new_p = (velocities - jnp.cross(omega[batch_segments], r)) * masses
    graph_result = jax.tree.map(lambda x: x, graph)
    graph_result.nodes['p'] = new_p

    if force_temperature:
        # Compute per-graph current temperature (in eV)
        temp0 = jraph_get_temperature(graph, n_dim=3, zero_drift=zero_drift, zero_rot=zero_rot)
        graph_result = jraph_force_temperature_fn(graph_result, temp0, n_dim=3, zero_drift=zero_drift, zero_rot=zero_rot)

    return graph_result
