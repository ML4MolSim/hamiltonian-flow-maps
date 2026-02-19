import jax
import unittest
import numpy as np
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import numpy as np
import hfm.utils as utils

from hfm.potentials.mueller_brown import MuellerBrownPotential
from hfm.simulation.nve_integrator import VelocityVerletIntegrator
from hfm.simulation.langevin_integrator import LangevinOperatorSplittingIntegrator
from hfm.utils import maxwell_boltzmann_distribution
from scipy.signal import find_peaks


class TestNVTSimulation(unittest.TestCase):
    def test_mueller_brown(self):
        rng = jax.random.PRNGKey(100)
        rng_mom, rng_langevin = jax.random.split(rng, 2)
        T = 23.0 / utils.kB
        T = jnp.array([T]).reshape(1, 1)

        vfm = MuellerBrownPotential()
        integrator = VelocityVerletIntegrator(vfm)
        lv_integrator = LangevinOperatorSplittingIntegrator(
            nve_integrator=integrator,
            temperature=T,
            integration_timestep=5e-3,
            time_constant=1.0)
        start_pos = jnp.array([-0.55828035, 1.44169]).reshape(1, 1, 2)
        start_mom = maxwell_boltzmann_distribution(rng_mom, vfm.masses, T, n_dim=2)
        xs, _, _, _ = lv_integrator(start_pos, start_mom, 800.0, rng_langevin)
        # we only started one simulation
        xs = xs[0]
        
        self.assertTrue(jnp.all(xs < 5.0))
        self.assertTrue(jnp.all(xs > -5.0))

        # find the peaks corresponding to the two minima
        def find_two_peaks(dim=0):
            hist, bin_edges = np.histogram(xs[:, :, dim], bins=60, density=True)
            peaks, _ = find_peaks(hist)

            top_two_indices = np.argsort(hist[peaks])[-2:]
            global_peaks = peaks[top_two_indices]
            return bin_edges[global_peaks]

        peak_x = find_two_peaks(dim=0)
        peak_y = find_two_peaks(dim=1)

        np.testing.assert_allclose(peak_x, [0.55, -0.558], atol=0.1)
        np.testing.assert_allclose(peak_y, [0.0, 1.441], atol=0.1)
