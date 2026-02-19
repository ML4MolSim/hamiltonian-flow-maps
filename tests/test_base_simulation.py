import unittest
import numpy as np
import jax
import jax.numpy as jnp

from hfm.utils import maxwell_boltzmann_distribution
from hfm.potentials.toy_potentials import HarmonicPotential
from hfm.simulation.nve_integrator import VelocityVerletIntegrator


class TestBaseSimulation(unittest.TestCase):
    def test_nested_simulation_same_as_non_nested(self):
        rng_mom, rng_int = jax.random.split(jax.random.PRNGKey(42))
        T = jnp.array([500.0]).reshape(1, 1)
        sim_len = 500.0

        system = HarmonicPotential()
        integrator = VelocityVerletIntegrator(system, integration_timestep=0.1, multistep_nested=True)

        start_pos = jnp.array([0.0]).reshape(1, 1, 1)
        start_mom = maxwell_boltzmann_distribution(rng_mom, system.masses, T, n_dim=1)

        xs_nested, ps_nested, vs_nested, fs_nested, _, _ = integrator._nested_call(
            start_pos, start_mom, sim_len, rng_int, {}, {}
        )
        xs_non_nested, ps_non_nested, vs_non_nested, fs_non_nested = integrator(
            start_pos, start_mom, sim_len, rng_int
        )

        np.testing.assert_allclose(xs_nested, xs_non_nested, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(ps_nested, ps_non_nested, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(vs_nested, vs_non_nested, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(fs_nested, fs_non_nested, rtol=1e-2, atol=1e-2)

    def test_different_intermediate_steps(self):
        rng_mom, rng_int = jax.random.split(jax.random.PRNGKey(42))
        T = jnp.array([500.0]).reshape(1, 1)
        sim_len = 500.0

        system = HarmonicPotential()
        integrator = VelocityVerletIntegrator(system, integration_timestep=0.1, multistep_nested=True)

        start_pos = jnp.array([0.0]).reshape(1, 1, 1)
        start_mom = maxwell_boltzmann_distribution(rng_mom, system.masses, T, n_dim=1)

        xs, ps, vs, fs = integrator(start_pos, start_mom, sim_len, rng_int)

        for intermediate_steps in [1, 2, 3, 100, 1000, 10000]:
            cur_xs, cur_ps, cur_vs, cur_fs = integrator(
                start_pos,
                start_mom,
                sim_len,
                rng_int,
                intermediate_steps=intermediate_steps,
            )
            np.testing.assert_allclose(xs, cur_xs, rtol=1e-2, atol=1e-2)
            np.testing.assert_allclose(ps, cur_ps, rtol=1e-2, atol=1e-2)
            np.testing.assert_allclose(vs, cur_vs, rtol=1e-2, atol=1e-2)
            np.testing.assert_allclose(fs, cur_fs, rtol=1e-2, atol=1e-2)

    def test_different_save_every(self):
        rng_mom, rng_int = jax.random.split(jax.random.PRNGKey(42))
        T = jnp.array([500.0]).reshape(1, 1)
        sim_len = 500.0

        system = HarmonicPotential()
        integrator = VelocityVerletIntegrator(system, integration_timestep=0.1)

        start_pos = jnp.array([0.0]).reshape(1, 1, 1)
        start_mom = maxwell_boltzmann_distribution(rng_mom, system.masses, T, n_dim=1)

        xs, ps, vs, fs = integrator(start_pos, start_mom, sim_len, rng_int)
        simulated_indices = np.arange(0, xs.shape[1]) + 1
        
        for save_every in [1, 2, 3, 923, 1000]:
            #print(f"Testing save_every: {save_every}")
            cur_xs, cur_ps, cur_vs, cur_fs = integrator(
                start_pos,
                start_mom,
                sim_len,
                rng_int,
                save_every=save_every,
            )
            should_keep = (simulated_indices % save_every == 0)
            #print(f"Should keep: {should_keep.shape}")
            
            np.testing.assert_allclose(xs[:, should_keep], cur_xs, rtol=1e-2, atol=1e-2)
            np.testing.assert_allclose(ps[:, should_keep], cur_ps, rtol=1e-2, atol=1e-2)
            np.testing.assert_allclose(vs[:, should_keep], cur_vs, rtol=1e-2, atol=1e-2)
            np.testing.assert_allclose(fs[:, should_keep], cur_fs, rtol=1e-2, atol=1e-2)


    def test_nested_simulation_same_as_non_nested_different_intermediate_steps_and_save_every(
        self,
    ):
        rng_mom, rng_int = jax.random.split(jax.random.PRNGKey(42))
        T = jnp.array([500.0]).reshape(1, 1)
        sim_len = 500.0

        system = HarmonicPotential()
        integrator = VelocityVerletIntegrator(system, integration_timestep=0.1, multistep_nested=True)

        start_pos = jnp.array([0.0]).reshape(1, 1, 1)
        start_mom = maxwell_boltzmann_distribution(rng_mom, system.masses, T, n_dim=1)

        for intermediate_steps in [100, 1000, 10000]:
            for save_every in [1, 2, 3, 100, 1000]:
                if save_every > intermediate_steps:
                    continue

                # print(
                #     f"Testing intermediate_steps: {intermediate_steps} and save_every: {save_every}"
                # )
                xs_nested, ps_nested, vs_nested, fs_nested, _, _ = integrator._nested_call(
                    start_pos,
                    start_mom,
                    sim_len,
                    rng_int,
                    {}, 
                    {},
                    save_every=save_every,
                )
                xs_non_nested, ps_non_nested, vs_non_nested, fs_non_nested = integrator(
                    start_pos,
                    start_mom,
                    sim_len,
                    rng_int,
                    intermediate_steps=intermediate_steps,
                    save_every=save_every,
                )

                np.testing.assert_allclose(
                    xs_nested, xs_non_nested, rtol=1e-2, atol=1e-2
                )
                np.testing.assert_allclose(
                    ps_nested, ps_non_nested, rtol=1e-2, atol=1e-2
                )
                np.testing.assert_allclose(
                    vs_nested, vs_non_nested, rtol=1e-2, atol=1e-2
                )
                np.testing.assert_allclose(
                    fs_nested, fs_non_nested, rtol=1e-2, atol=1e-2
                )
