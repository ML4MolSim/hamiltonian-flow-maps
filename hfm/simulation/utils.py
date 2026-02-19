from functools import partial
import ase
import jax
import numpy as np

import jax.numpy as jnp
import numpy as np
import numpy as np
from jax import jit

from jax import vmap
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

from hfm import utils
from hfm.utils import kabsch_algorithm


def compute_dihedral(p1, p2, p3, p4):
    """
    Compute the dihedral (torsion) angle between four points in 3D space.

    Parameters:
    p1, p2, p3, p4: numpy arrays of shape (3,), representing atomic coordinates.

    Returns:
    Dihedral angle in degrees.
    """
    # Define vectors
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # Compute normal vectors to planes
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)

    # Normalize normal vectors
    n1 /= jnp.linalg.norm(n1)
    n2 /= jnp.linalg.norm(n2)

    # Compute the unit vector along b2
    b2 /= jnp.linalg.norm(b2)

    # Compute the dihedral angle
    x = jnp.dot(n1, n2)
    y = jnp.dot(jnp.cross(n1, n2), b2)
    angle = jnp.arctan2(y, x)
    # angle = jnp.arccos(x)

    # Convert from radians to degrees
    return jnp.degrees(angle)


compute_dihedral_batch = vmap(compute_dihedral, in_axes=(0, 0, 0, 0))


def estimate_velocities_from_positions(positions, dt):
    """
    Args:
        positions (Array): (num_frames, num_atoms, dim)
        dt (float): time step in fs between frames
    """

    displacements = np.diff(positions, axis=0)
    return displacements / dt


def calculate_power_spectrum(velocities, dt):
    """
    Args:
        velocities (Array): (num_frames, num_atoms, dim).
        dt (float): time step in fs between frames

    Returns:
        frequencies: The frequencies, (num_frames // 2)
        power_spectrum: The power spectrum, (num_frames // 2)

    """
    dt = dt / ase.units.fs  # assume dt is in fs

    num_frames = velocities.shape[0]
    
    vel_fft = np.fft.fft(velocities, axis=0)
    
    power_per_component = np.abs(vel_fft)**2
    
    power_per_atom = np.sum(power_per_component, axis=2)
    
    power_spectrum = np.mean(power_per_atom, axis=1)
    
    freqs = np.fft.fftfreq(num_frames, d=dt) 
    freqs = freqs * 33356.4095198152 # Frequency in cm^-1

    return freqs[freqs > 0], power_spectrum[freqs > 0] / power_spectrum[freqs > 0].max()


def calculate_power_spectrum_sgdml(velocities, dt):
    dt = dt / ase.units.fs  # assume dt is in fs
    V = velocities.reshape(velocities.shape[0], -1).T  # (num_dof, num_steps)
    n_steps = V.shape[1]

    # mean velocity auto-correlation for all degrees of freedom
    vac2 = [np.correlate(v, v, 'full') for v in V]
    vac2 /= np.linalg.norm(vac2, axis=1)[:, None]
    vac2 = np.mean(vac2, axis=0)

    # power spectrum (phonon density of states)
    pdos = np.abs(np.fft.fft(vac2))**2
    pdos /= np.linalg.norm(pdos) / 2 # spectrum is symmetric

    freq = np.fft.fftfreq(2*n_steps-1, dt)
    freq = freq * 33356.4095198152 # Frequency in cm^-1

    return freq[:n_steps], pdos[:n_steps]


@jit
def calculate_power_spectrum_sgdml_jax(velocities, dt):
    dt = dt / ase.units.fs 

    V = velocities.reshape(velocities.shape[0], -1).T
    _, n_steps = V.shape
    
    # Target length to match np.correlate('full')
    N = 2 * n_steps - 1
    
    # Pad only on the right to reach size N
    # We need to add N - n_steps = n_steps - 1 zeros
    Vpad = jnp.pad(V, ((0, 0), (0, n_steps - 1))) 

    # FFT of padded signals
    F = jnp.fft.fft(Vpad)

    # Autocorrelation (Wiener-Khinchin theorem)
    # result has length N = 2*n_steps - 1
    ac = jnp.fft.ifft(F * jnp.conj(F)).real 

    # Normalize
    norms = jnp.linalg.norm(ac, axis=1, keepdims=True)
    ac_norm = ac / norms

    vac = jnp.mean(ac_norm, axis=0)

    # Power spectrum
    # vac has length N, so pdos has length N
    pdos = jnp.abs(jnp.fft.fft(vac))**2
    pdos = pdos / (jnp.linalg.norm(pdos) / 2.0)

    # Frequency axis
    # Now N matches the length of pdos exactly
    freq = jnp.fft.fftfreq(N, dt)
    freq = freq * 33356.4095198152

    return freq[:n_steps], pdos[:n_steps]


def smoothen_spectrum(power, window_length: int = 50, polyorder: int = 1):
    """ Use savitzky-Golay filter. Default values might need to be adjusted.
    Args:
        power: power spectrum, (num_frames // 2, )
        window_length: window length for filter
        polyorder: polynomial order for fitting.

    """
    return savgol_filter(power, window_length, polyorder)


def plot_spectrum(positions, dt, velocities=None, ax=None, window_length=50, label="Power Spectrum", alpha=0.6, linestyle="-", algorithm="sgdml_jax", filter="gaussian", sigma=50, log=True, **kwargs):
    if ax is None:
        ax = plt.gca()
    
    if velocities is None:
        velocities = estimate_velocities_from_positions(positions=positions, dt=dt)
    
    if algorithm == "sgdml_jax":
        freq, spectrum = calculate_power_spectrum_sgdml_jax(velocities=velocities, dt=dt)
    elif algorithm == "sgdml":
        freq, spectrum = calculate_power_spectrum_sgdml(velocities=velocities, dt=dt)
    elif algorithm == "fft_max":
        freq, spectrum = calculate_power_spectrum(velocities=velocities, dt=dt)
    else:
        raise ValueError(f"Unknown algorithm {algorithm} for power spectrum calculation.")

    if filter == "savitzky":
        if window_length < len(spectrum):
            spectrum = smoothen_spectrum(spectrum, window_length=window_length)
        elif log:
            print("Skipping smoothing because trajectory length is shorter than window length.")
    elif filter == "gaussian":
        spectrum = gaussian_filter1d(spectrum, sigma=sigma)
    else:
        raise ValueError(f"Unknown filter {filter} for power spectrum smoothing.")

    ax.plot(freq, spectrum, label=label, alpha=alpha, linestyle=linestyle, **kwargs)

    # compute xlim to the right
    spectrum_threshold = 0.01 * spectrum.max()
    above_threshold = freq[spectrum > spectrum_threshold]
    return above_threshold[-1]


def align_positions(xs):
    aligned_xs = np.copy(xs)
    aligned_xs = aligned_xs - np.mean(aligned_xs, axis=1, keepdims=True) # remove com
    ref = np.array(aligned_xs[0])
    for i in range(aligned_xs.shape[0]):
        aligned_xs[i] = kabsch_algorithm(aligned_xs[i], ref)
    return aligned_xs


def convert_to_asetraj(xs, atomic_numbers, align_xs=True):
    if align_xs:
        xs = align_positions(xs)

    return [ase.Atoms(atomic_numbers.reshape(-1), positions=xs[i]) for i in range(xs.shape[0])]


def running_average(data, window_size):
    """Compute the running average of a 1D array."""
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    if window_size > len(data):
        raise ValueError("Window size must not be larger than the data length.")
    
    return jnp.convolve(data, jnp.ones(window_size) / window_size, mode='valid')


@partial(jax.jit, static_argnames=("get_epot", "running_avg_window", "return_ekin_epot", "batch_size"))
def get_total_energy(
    xs,
    ps,
    get_epot,
    masses,
    running_avg_window=1000,
    return_ekin_epot=False,
    batch_size=256,
):
    """
    Compute total energy with batching using JAX and lax.scan.

    Args:
        xs: (N, ...) array of positions
        ps: (N, ...) array of momenta
        get_epot: function(xs, ps) -> potential energies
        masses: array of particle masses
        running_avg_window: window size for running average
        return_ekin_epot: whether to also return (ekin, epot)
        batch_size: batch size for inference
    """
    def pad_to_multiple(x, multiple):
        N = x.shape[0]
        pad = (multiple - (N % multiple)) % multiple
        pad_width = [(0, pad)] + [(0,0)] * (x.ndim - 1)
        return jnp.pad(x, pad_width), N

    xs, N = pad_to_multiple(xs, batch_size)
    ps, _ = pad_to_multiple(ps, batch_size)

    def body_fn(carry, i):
        start = i * batch_size

        xb = jax.lax.dynamic_slice_in_dim(xs, start, batch_size, axis=0)
        pb = jax.lax.dynamic_slice_in_dim(ps, start, batch_size, axis=0)

        ekin_b = utils.kinetic_energy(pb, masses).reshape(-1)
        epot_b = get_epot(xb, pb).reshape(-1)

        return carry, (ekin_b, epot_b)

    num_batches = xs.shape[0] // batch_size
    _, (ekins, epots) = jax.lax.scan(body_fn, None, jnp.arange(num_batches))

    # concatenate batched results
    ekin = jnp.concatenate(ekins, axis=0)[:N]
    epot = jnp.concatenate(epots, axis=0)[:N]
    etot = ekin + epot

    if return_ekin_epot:
        return (
            running_average(etot, running_avg_window),
            running_average(ekin, running_avg_window),
            running_average(epot, running_avg_window),
        )
    else:
        return running_average(etot, running_avg_window)
