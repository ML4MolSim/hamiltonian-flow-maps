import pickle
import pathlib
import unittest
import numpy as np
import jax
import jax.numpy as jnp

from hfm import utils
from hfm.backbones.mlp import MLPBackbone
from hfm.potentials.neural_force_field import NeuralForceField
from hfm.utils import maxwell_boltzmann_distribution
from hfm.potentials.toy_potentials import HarmonicPotential
from hfm.simulation.nve_integrator import VelocityVerletIntegrator


class TestNVEIntegrator(unittest.TestCase):
    def test_energy_conservation(self):
        rng_mom, rng_int = jax.random.split(jax.random.PRNGKey(42))       
        T = jnp.array([500.0]).reshape(1, 1)
        sim_len = 500.0 

        system = HarmonicPotential()
        integrator = VelocityVerletIntegrator(system, integration_timestep=.1)

        start_pos = jnp.array([0.0]).reshape(1, 1, 1)
        start_mom = maxwell_boltzmann_distribution(rng_mom, system.masses, T, n_dim=1)

        xs, ps, _, _ = integrator(start_pos, start_mom, sim_len, rng_int)
        epots = system.compute_epot(xs, ps).reshape(-1)
        ekins = utils.kinetic_energy(ps, system.masses).reshape(-1)
        etots = epots + ekins

        np.testing.assert_allclose(etots, etots[0], rtol=1e-2, atol=1e-2)

    def test_neural_force_field(self):
        rng_mom, rng_int = jax.random.split(jax.random.PRNGKey(42))       
        T = jnp.array([5000.0]).reshape(1, 1)
        sim_len = 10.0 

        package_dir = pathlib.Path(__file__).parent.resolve()

        with open(package_dir / "params_ho.pkl", "rb") as f:
            params = pickle.load(f)
        model = MLPBackbone()

        ff = NeuralForceField(
            model=model,
            params=params,
            masses=HarmonicPotential().masses,
            atomic_numbers=jnp.zeros(HarmonicPotential().masses.shape, dtype=int),
        )
        integrator = VelocityVerletIntegrator(ff, integration_timestep=1e-3)
        start_pos = jnp.array([0.0]).reshape(1, 1, 1)
        start_mom = maxwell_boltzmann_distribution(rng_mom, ff.masses, T, n_dim=1)
        xs, ps, _, _ = integrator(start_pos, start_mom, sim_len, rng_int)

        r_squared = (xs**2 + ps**2).reshape(-1)
        np.testing.assert_allclose(r_squared, r_squared[0], rtol=1e-2, atol=1e-2)
