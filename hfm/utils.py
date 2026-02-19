import jax.numpy as jnp
from jax import vmap
import jax.random as jr
import numpy as np
import jax.numpy as jnp
from jax import vmap

kB = 8.617333262145e-5  # eV/K
eps_temp = 1e-8


def verify_shapes(masses=None, atomic_numbers=None, positions=None, momenta=None, forces=None, energies=None, time=None, temperature=None):
    if masses is not None:
        assert masses.ndim == 3, f'masses is expected to have shape (batch_dim, num_atoms, 1). received {masses.shape=}'
        assert masses.shape[2] == 1, f'masses is expected to have shape (batch_dim, num_atoms, 1). received {masses.shape=}'
    
    if atomic_numbers is not None:
        assert atomic_numbers.ndim == 3, f'atomic_numbers is expected to have shape (batch_dim, num_atoms, 1). received {atomic_numbers.shape=}'
        assert atomic_numbers.shape[2] == 1, f'atomic_numbers is expected to have shape (batch_dim, num_atoms, 1). received {atomic_numbers.shape=}'
    
    if positions is not None:
        assert positions.ndim == 3, f'positions is expected to have shape (batch_dim, num_atoms, n_dim). received {positions.shape=}'
    
    if momenta is not None:
        assert momenta.ndim == 3, f'momenta is expected to have shape (batch_dim, num_atoms, n_dim). received {momenta.shape=}'
    
    if forces is not None:
        assert forces.ndim == 3, f'forces is expected to have shape (batch_dim, num_atoms, n_dim). received {forces.shape=}'

    if energies is not None:
        assert energies.ndim == 2, f'energies is expected to have shape (batch_dim, 1). received {energies.shape=}'
        assert energies.shape[1] == 1, f'energies is expected to have shape (batch_dim, 1). received {energies.shape=}'

    if time is not None:
        assert time.ndim == 2, f'time is expected to have shape (batch_dim, 1). received {time.shape=}'
        assert time.shape[1] == 1, f'time is expected to have shape (batch_dim, 1). received {time.shape=}'

    if temperature is not None:
        assert temperature.ndim == 2, f'temperature is expected to have shape (batch_dim, 1). received {temperature.shape=}'
        assert temperature.shape[1] == 1, f'temperature is expected to have shape (batch_dim, 1). received {temperature.shape=}'


def get_dof(masses, n_dim=3, zero_drift: bool = False, zero_rot: bool = False):
    verify_shapes(masses=masses)
    
    num_atoms = masses.shape[1]
    ndof = num_atoms * n_dim

    # For the single particle one can not remove drift and rotation.
    if num_atoms == 1:
        return ndof

    if zero_drift:
        ndof -= n_dim
    # Rotations only exist in 2D and higher dimensions.
    if zero_rot and n_dim > 1:
        ndof -= n_dim

    return ndof

def kinetic_energy(momenta, masses):
    verify_shapes(masses=masses)
    
    return 0.5 * jnp.sum((momenta**2) / masses, axis=(1, 2)).reshape(-1, 1)

def get_temperature_from_kin_energy(ekin, masses, unit="K", n_dim=3, zero_drift: bool = False, zero_rot: bool = False):
    verify_shapes(masses=masses, energies=ekin)
    
    ndof = get_dof(masses, n_dim=n_dim, zero_drift=zero_drift, zero_rot=zero_rot)
    temp_K = 2 * ekin / (ndof * kB)
    return convert_temperature(temp_K, unit_from="K", unit_to=unit)

def get_kin_energy_from_temperature(temperature, masses, n_dim=3, zero_drift: bool = False, zero_rot: bool = False):
    ndof = get_dof(masses, n_dim=n_dim, zero_drift=zero_drift, zero_rot=zero_rot)
    e_kin = 0.5 * ndof * convert_temperature(temperature, unit_from="K", unit_to="eV")
    return e_kin

def get_temperature(momenta, masses, unit="K", n_dim=3, zero_drift: bool = False, zero_rot: bool = False):
    verify_shapes(masses=masses, momenta=momenta)
    
    ekin = kinetic_energy(momenta, masses)
    ndof = get_dof(masses, n_dim=n_dim, zero_drift=zero_drift, zero_rot=zero_rot)
    temp_K = 2 * ekin / (ndof * kB)
    return convert_temperature(temp_K, unit_from="K", unit_to=unit)

def convert_temperature(temperature, unit_from="K", unit_to="eV"):
    verify_shapes(temperature=temperature)

    if unit_from == "K" and unit_to == "eV":
        return temperature * kB
    elif unit_from == "eV" and unit_to == "K":
        return temperature / kB
    elif unit_from == unit_to:
        return temperature
    else:
        raise ValueError("Unit must be 'K' or 'eV'")

def force_temperature_fn(momenta, masses, temperature, unit="K", n_dim=3, zero_drift: bool = False, zero_rot: bool = False):
    verify_shapes(masses=masses, momenta=momenta, temperature=temperature)

    target_temp = convert_temperature(temperature, unit_from=unit, unit_to="eV")
    ekin = kinetic_energy(momenta, masses)
    ndof = get_dof(masses, n_dim=n_dim, zero_drift=zero_drift, zero_rot=zero_rot)
    current_temp = 2 * ekin / ndof
    scale = jnp.where(temperature > eps_temp, target_temp / current_temp, 0.0)
    scale = jnp.sqrt(scale)[..., None]

    return momenta * scale

# See refrence implementation in ASE:
# https://gitlab.jsc.fz-juelich.de/kesselheim1/ase/-/blob/master/ase/md/velocitydistribution.py
def maxwell_boltzmann_distribution(key, masses, temperature, unit="K", n_dim=3, force_temperature: bool = True):
    verify_shapes(masses=masses, temperature=temperature)

    temp_eV = convert_temperature(temperature, unit_from=unit, unit_to="eV")
    N = masses.shape[1]
    B = temperature.shape[0]
    xi = jr.normal(key, shape=(B, N, n_dim))
    momenta = xi * jnp.sqrt(masses * temp_eV[..., None])
    
    # Rescale momenta to exactly match `temperature`.
    if force_temperature:
        momenta = force_temperature_fn(momenta, masses, temperature=temp_eV, unit="eV", n_dim=n_dim)
    
    return momenta

def remove_center(x):
    verify_shapes(positions=x)

    assert x.ndim == 3, f'array is expected to have shape (batch_dim, num_atoms, n_dim). received {x.shape=}'
    num_atoms = x.shape[-2]
    
    if num_atoms == 1:
        return x
    else:
        return x - x.mean(axis=-2, keepdims=True)

def get_global_momenta_drift(momenta, masses):
    verify_shapes(momenta=momenta, masses=masses)
    return jnp.sum(momenta, axis=1)  # (B, n_dim)

def stationary(momenta, masses, force_temperature: bool = True, zero_drift: bool = False, zero_rot: bool = False):
    verify_shapes(momenta=momenta, masses=masses)
    
    num_atoms = masses.shape[1]
    n_dim = momenta.shape[-1]
    
    if num_atoms == 1 or n_dim == 1:
        return momenta
    
    total_p = jnp.sum(momenta, axis=1)  # (B, n_dim)
    total_mass = jnp.sum(masses, axis=1)  # (B, 1)
    v0 = total_p / total_mass  # (B, n_dim)
    new_momenta = momenta - v0[:, None, :] * masses  # (B, N, n_dim)

    # Rescale momenta to match the original temperature of the input momenta.
    if force_temperature:
        temp0 = get_temperature(momenta, masses, zero_drift=zero_drift, zero_rot=zero_rot)
        new_momenta = force_temperature_fn(new_momenta, masses, temp0, zero_drift=zero_drift, zero_rot=zero_rot)
    
    return new_momenta

def inertia_tensor(positions, masses):
    verify_shapes(positions=positions, masses=masses)
    masses = masses.squeeze(-1)  # (B, N)
    x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
    
    I11 = jnp.sum(masses * (y**2 + z**2), axis=1)
    I22 = jnp.sum(masses * (x**2 + z**2), axis=1)
    I33 = jnp.sum(masses * (x**2 + y**2), axis=1)
    I12 = -jnp.sum(masses * x * y, axis=1)
    I13 = -jnp.sum(masses * x * z, axis=1)
    I23 = -jnp.sum(masses * y * z, axis=1)

    I = jnp.stack([
        jnp.stack([I11, I12, I13], axis=-1),
        jnp.stack([I12, I22, I23], axis=-1),
        jnp.stack([I13, I23, I33], axis=-1)
    ], axis=-2)  # (B, 3, 3)
    return I

def global_angular_momentum_3d(positions, momenta, masses):
    verify_shapes(positions=positions, momenta=momenta, masses=masses)

    com = jnp.sum(positions * masses, axis=1) / jnp.sum(masses, axis=1)
    r = positions - com[:, None, :]  # (B, N, 3)

    # Calculate the angular momentum
    return jnp.sum(jnp.cross(r, momenta), axis=1)  # (B, 3)

def remove_global_rotation_3d(positions, momenta, masses, target_L=0):
    """Remove global rotations in 3D.
    Args:
        positions (Array): (B, N, 3)
        momenta (Array): (B, N, 3)
        masses (Array): (B, N, 1)
        target_L (Array): (B, 3)
    Returns:
        Array: (B, N, 3)
    """
    verify_shapes(positions=positions, momenta=momenta, masses=masses)
    velocities = momenta / masses

    # Remove the com from the positions.
    com = jnp.sum(positions * masses, axis=1) / jnp.sum(masses, axis=1)
    r = positions - com[:, None, :]  # (B, N, 3)

    # Calculate the angular momentum and the moments of inertia.
    L = jnp.sum(jnp.cross(r, masses * velocities), axis=1)  # (B, 3)
    I = inertia_tensor(r, masses)  # (B, 3, 3)

    def solve_omega(Ii, Li):
        return jnp.linalg.solve(Ii + jnp.eye(3) * 1e-8, Li)

    omega = vmap(solve_omega)(I, L - target_L)  # (B, 3)

    corrected_momenta = (velocities - jnp.cross(omega[:, None, :], r)) * masses  # (B, N, 3)
    return corrected_momenta  # (B, N, 3)

def remove_global_rotation_2d(positions, momenta, masses):
    """Remove global rotations in 2D.
    Args:
        positions (Array): (B, N, 2)
        momenta (Array): (B, N, 2)
        masses (Array): (B, N, 1)
    Returns:
        Array: (B, N, 2)
    """
    verify_shapes(positions=positions, momenta=momenta, masses=masses)
    total_mass = jnp.sum(masses, axis=-2, keepdims=True) # (B, 1)
    
    velocities = momenta / masses # (B, N, 2)

    com_pos = jnp.sum(masses * positions, axis=-2, keepdims=True) / total_mass  # (B, 1, 3)
    
    pos_com = positions - com_pos  # (B, N, 3)

    # Take the x and y entries and sum over all atoms to get angular momentum.
    angular_momentum = jnp.sum(
        jnp.squeeze(masses, axis=-1) * (pos_com[..., 0] * velocities[..., 1] - pos_com[..., 1] * velocities[..., 0]),
        axis=-1
    ) # (B, )

    # Calculate Moment of Inertia along z.
    moment_of_inertia = jnp.sum(
        jnp.squeeze(masses, axis=-1) * pos_com * jnp.sum(pos_com**2, axis=-1),
        axis=-1
    )  # (B, )

    # Determine Angular Velocity.
    angular_velocity = jnp.where(
        jnp.abs(moment_of_inertia) > 1e-4,
        angular_momentum / moment_of_inertia,
        0.0
    )  # (B, )

    # Subtract the rotational velocity component from each particle
    omega_z = jnp.expand_dims(angular_velocity, axis=-1)  # (B, 1)
    
    vel_rot_x = -omega_z * pos_com[..., 1]
    vel_rot_y =  omega_z * pos_com[..., 0]
    
    # Stack the x and y components to get v_rot of shape (B, N, 2)
    vel_rot = jnp.stack((vel_rot_x, vel_rot_y), axis=-1)
    
    final_velocities = velocities - vel_rot
    
    return final_velocities * masses  # (B, N, 2)

def apply_constraints(positions, momenta, masses, zero_drift: bool = False, zero_rot: bool = False):
    verify_shapes(positions=positions, momenta=momenta, masses=masses)

    # Just zero-centering here, not COM removal
    positions = remove_center(positions)
    momenta = zero_rotation(positions, momenta, masses, zero_drift=zero_drift, zero_rot=zero_rot, force_temperature=False)
    momenta = stationary(momenta, masses, zero_drift=zero_drift, zero_rot=True, force_temperature=False)

    return positions, momenta

def zero_rotation(positions, momenta, masses, force_temperature: bool = True, zero_drift: bool = False, zero_rot: bool = False):
    verify_shapes(positions=positions, momenta=momenta, masses=masses)
    
    num_atoms = masses.shape[1]
    n_dim = positions.shape[-1]
    
    if num_atoms == 1:
        return momenta
    
    if n_dim == 3:
        corrected_momenta = remove_global_rotation_3d(positions, momenta, masses)
    elif n_dim == 2:
        corrected_momenta = remove_global_rotation_2d(positions, momenta, masses)
    elif n_dim == 1:
        corrected_momenta = momenta
    else:
        raise ValueError(
            f'Number of dimensions must be 1, 2 or 3. received {n_dim=}.'
        )
    
    # Rescale momenta to match the original temperature of the input momenta.
    if force_temperature:
        temp0 = get_temperature(momenta, masses, zero_drift=zero_drift, zero_rot=zero_rot)
        corrected_momenta = force_temperature_fn(corrected_momenta, masses, temp0, zero_drift=zero_drift, zero_rot=zero_rot)  # (B, n_dim)

    return corrected_momenta  # (B, n_dim)

def zero_torque(positions, forces):
    verify_shapes(positions=positions, forces=forces)

    n_dim = positions.shape[-1]
    num_atoms = positions.shape[1]

    if num_atoms == 1:
        return forces
    
    if n_dim == 3:
        return remove_global_rotation_3d(positions, forces, jnp.ones((1, num_atoms)))
    elif n_dim == 2:
        return remove_global_rotation_2d(positions, forces, jnp.ones((1, num_atoms)))
    elif n_dim == 1:
        return forces
    else:
        raise ValueError(
            f'Number of dimensions must be 1, 2 or 3. received {n_dim=}.'
        )

def global_torque(positions, forces):
    verify_shapes(positions=positions, forces=forces)

    com = jnp.mean(positions, axis=1, keepdims=True)
    r = positions - com

    tau = jnp.sum(jnp.cross(r, forces), axis=1)  # (B, 3)
    return tau

def kabsch_algorithm(P, Q, return_rotation: bool = False):
    """
    Perform the Kabsch algorithm to find the optimal rotation matrix.

    Parameters:
        P (numpy.ndarray): A 2D array of shape (N, 3) representing the current coordinates.
        Q (numpy.ndarray): A 2D array of shape (N, 3) representing the reference coordinates.

    Returns:
        R (numpy.ndarray): The optimal rotation matrix of shape (3, 3).
    """
    # Step 1: Compute the covariance matrix
    H = np.dot(P.T, Q)

    # Step 2: Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)

    # Step 3: Compute the rotation matrix
    R = np.dot(Vt.T, U.T)

    # Step 4: Ensure a proper rotation (determinant = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    if return_rotation is True:
        return P@R.T, R
    else:
        return P@R.T

def kabsch_algorithm_jax(P, Q, return_rotation: bool = False):
    """
    Perform the Kabsch algorithm to find the optimal rotation matrix.

    Parameters:
        P (numpy.ndarray): A 2D array of shape (N, 3) representing the current coordinates.
        Q (numpy.ndarray): A 2D array of shape (N, 3) representing the reference coordinates.

    Returns:
        R (numpy.ndarray): The optimal rotation matrix of shape (3, 3).
    """
    # Step 1: Compute the covariance matrix
    H = jnp.dot(P.T, Q)

    # Step 2: Perform Singular Value Decomposition (SVD)
    U, S, Vt = jnp.linalg.svd(H)

    # Step 3: Compute the rotation matrix
    R = jnp.dot(Vt.T, U.T)

    # Step 4: Ensure a proper rotation (determinant = +1)
    d = jnp.linalg.det(R)
    flip_mask = (d < 0.0).reshape(-1, 1)
    flip_matrix = jnp.array([[1., 0., 0.], [0., 1., 0], [0., 0., -1.]])
    Vt = jnp.where(flip_mask, jnp.matmul(flip_matrix, Vt), Vt)
    R = jnp.dot(Vt.T, U.T)

    if return_rotation is True:
        return P@R.T, R
    else:
        return P@R.T

def estimate_velocities_from_positions(positions, dt):
    """
    Args:
        positions (Array): (num_frames, num_atoms, n_dim)
        dt (float): time step in fs between frames
    """
    verify_shapes(positions=positions)
    
    displacements = jnp.diff(positions, axis=0)
    return displacements / dt

def rotation_matrix_2d(theta):
    return jnp.array([
        [jnp.cos(theta), -jnp.sin(theta)],
        [jnp.sin(theta), jnp.cos(theta)]
    ])

def random_rotation_2d(key, num=1):
    theta = jr.uniform(key, shape=(num,), minval=0, maxval=2 * jnp.pi)
    return vmap(rotation_matrix_2d, in_axes=0, out_axes=0)(theta)
