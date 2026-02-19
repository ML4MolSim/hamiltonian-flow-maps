import unittest
import jax.random as random
import jax.numpy as jnp
import hfm.simulation.utils as utils


class TestSpectra(unittest.TestCase):
    def test_sgdml_vs_reference(self):
        key = random.PRNGKey(0)
        traj = random.normal(key, (100, 5, 3))  # random vel for 5 particles

        freq, pdos = utils.calculate_power_spectrum_sgdml(velocities=traj, dt=0.5)
        freq_JAX, pdos_jax = utils.calculate_power_spectrum_sgdml_jax(velocities=traj, dt=0.5)

        self.assertTrue(jnp.allclose(freq_JAX, freq))
        self.assertTrue(jnp.allclose(pdos, pdos_jax))
