import unittest
import jax
import jax.numpy as jnp
from tests.test_utils.ff import DummyPotential
from hfm import utils
from hfm.potentials.n_body_gravity import NBodyGravityPotential
from hfm.simulation.integration_filters import (
    CoupledConservationFilter,
    PreserveAngularMomentumFilter,
    EnergyConservationFilter,
    RandomRotationFilter,
    RemoveDriftFilterFlashMD,
)
from hfm.simulation.nve_integrator import EulerIntegrator, VelocityVerletIntegrator
from hfm.utils import global_angular_momentum_3d, kinetic_energy


class TestFilters(unittest.TestCase):
    def test_simulation_with_filters(self):
        # first we need to check if we can load ethanol
        start_pos = jax.random.normal(jax.random.PRNGKey(42), (2, 9, 3))
        start_mom = jax.random.normal(jax.random.PRNGKey(42), (2, 9, 3))

        potential = DummyPotential(masses=jnp.ones((1, 9, 1)))
        integrator = VelocityVerletIntegrator(potential, integration_timestep=1)

        integrator.add_integration_filter(PreserveAngularMomentumFilter())
        integrator.add_integration_filter(RemoveDriftFilterFlashMD())
        integrator.add_integration_filter(EnergyConservationFilter(potential))
        integrator.add_integration_filter(RandomRotationFilter())

        end_pos, end_mom, _, _ = integrator(
            start_pos, start_mom, 10, jax.random.PRNGKey(42)
        )

        # we perform 10 steps
        assert end_pos.shape == (2, 10, 9, 3), f"end_pos.shape: {end_pos.shape}"
        assert end_mom.shape == (2, 10, 9, 3), f"end_mom.shape: {end_mom.shape}"

    def test_zero_rot(self):
        rng = jax.random.PRNGKey(42)
        rng_pos, rng_mom = jax.random.split(rng)

        start_pos = jax.random.normal(rng_pos, (2, 10, 3))
        start_mom = jax.random.normal(rng_mom, (2, 10, 3))
        masses = jnp.ones((1, 10, 1))

        p_new = utils.zero_rotation(start_pos, start_mom, masses, force_temperature=False)
        end_L = global_angular_momentum_3d(
            start_pos, p_new, masses
        )

        assert jnp.allclose(
            end_L, jnp.zeros_like(end_L), atol=1e-5
        ), f"Angular momentum not zeroed: {p_new}"

    def test_angular_momentum_preservation(self):
        rng = jax.random.PRNGKey(42)
        rng_pos, rng_mom = jax.random.split(rng)

        start_pos = jax.random.normal(rng_pos, (2, 10, 3))
        start_mom = jax.random.normal(rng_mom, (2, 10, 3))

        potential = NBodyGravityPotential()
        # use non-symplectic integrator to break angular momentum conservation
        integrator = EulerIntegrator(potential, integration_timestep=0.1)  
        integrator.add_integration_filter(PreserveAngularMomentumFilter())

        end_pos, end_mom, _, _ = integrator(
            start_pos, start_mom, 1, jax.random.PRNGKey(42)
        )

        start_L = global_angular_momentum_3d(
            start_pos, start_mom, potential.masses
        )
        end_L = global_angular_momentum_3d(
            end_pos[:, -1], end_mom[:, -1], potential.masses
        )

        assert jnp.allclose(
            start_L, end_L, atol=1e-5
        ), f"Initial angular momentum not equal to final angular momentum: {start_L} vs {end_L}"

    def test_angular_momentum_zero_energy_change(self):
        rng = jax.random.PRNGKey(42)
        rng_pos, rng_mom = jax.random.split(rng)

        start_pos = jax.random.normal(rng_pos, (2, 10, 3))
        start_mom = jax.random.normal(rng_mom, (2, 10, 3))

        potential = NBodyGravityPotential()
        # use non-symplectic integrator to break angular momentum & energy conservation
        integrator = EulerIntegrator(potential, integration_timestep=0.01)  
        integrator.add_integration_filter(PreserveAngularMomentumFilter())
        integrator.add_integration_filter(EnergyConservationFilter(potential))

        end_pos, end_mom, _, _ = integrator(
            start_pos, start_mom, 0.1, jax.random.PRNGKey(42)
        )

        start_L = global_angular_momentum_3d(
            start_pos, start_mom, potential.masses
        )
        end_L = global_angular_momentum_3d(
            end_pos[:, -1], end_mom[:, -1], potential.masses
        )

        start_energy = kinetic_energy(
            start_mom, potential.masses
        ) + potential.compute_epot(start_pos)
        end_energy = kinetic_energy(
            end_mom[:, -1], potential.masses
        ) + potential.compute_epot(end_pos[:, -1])

        assert jnp.allclose(
            start_energy, end_energy, atol=1e-4
        ), f"Initial energy not equal to final energy: {start_energy} vs {end_energy}"

        assert jnp.allclose(
            start_L, end_L, atol=1e-4
        ), f"Initial angular momentum not equal to final angular momentum: {start_L} vs {end_L}"

    def test_coupled_conservation(self):
        rng = jax.random.PRNGKey(42)
        rng_pos, rng_mom = jax.random.split(rng)

        start_pos = jax.random.normal(rng_pos, (2, 10, 3))
        start_mom = jax.random.normal(rng_mom, (2, 10, 3))

        potential = NBodyGravityPotential()
        # use non-symplectic integrator to break angular momentum & energy conservation
        integrator = EulerIntegrator(potential, integration_timestep=0.01)  
        integrator.add_integration_filter(CoupledConservationFilter(potential))

        end_pos, end_mom, _, _ = integrator(
            start_pos, start_mom, 0.1, jax.random.PRNGKey(42)
        )

        start_L = global_angular_momentum_3d(
            start_pos, start_mom, potential.masses
        )
        end_L = global_angular_momentum_3d(
            end_pos[:, -1], end_mom[:, -1], potential.masses
        )

        start_energy = kinetic_energy(
            start_mom, potential.masses
        ) + potential.compute_epot(start_pos)
        end_energy = kinetic_energy(
            end_mom[:, -1], potential.masses
        ) + potential.compute_epot(end_pos[:, -1])

        assert jnp.allclose(
            start_energy, end_energy, atol=1e-4
        ), f"Initial energy not equal to final energy: {start_energy} vs {end_energy}"

        assert jnp.allclose(
            start_L, end_L, atol=1e-4
        ), f"Initial angular momentum not equal to final angular momentum: {start_L} vs {end_L}"

    def test_angular_momentum_convervation_does_not_change_drift(self):
        rng = jax.random.PRNGKey(42)
        rng_pos, rng_mom = jax.random.split(rng)

        start_pos = jax.random.normal(rng_pos, (2, 10, 3))
        start_mom = jax.random.normal(rng_mom, (2, 10, 3))
        masses = jnp.ones((1, 10, 1))

        # make them drift free
        start_drift = start_mom.sum(axis=1)

        corrected_mom = utils.remove_global_rotation_3d(start_pos, start_mom, masses, 1)
        end_drift = corrected_mom.sum(axis=1)

        self.assertTrue(jnp.allclose(
            start_drift, end_drift, atol=1e-5
        ), f"Drift changed after angular momentum removal: {start_drift} vs {end_drift}")

        self.assertFalse(jnp.allclose(start_mom, corrected_mom, atol=1e-5), "Momenta did not change after angular momentum removal")
