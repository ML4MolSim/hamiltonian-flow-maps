import jax
import unittest
import jax
import ase
from hfm.datasets.in_memory_data_module import InMemoryDataModule
from hfm.simulation.potential import Potential
import jax.numpy as jnp
import hfm.simulation.metrics as metrics


class DummyPotential(Potential):
    def compute_epot(self, x, p=None) -> jnp.ndarray:
        return jnp.zeros((x.shape[0],))

    def compute_force(self, x, p=None) -> jnp.ndarray:
        return jnp.zeros_like(x)
    
    def compute_force_and_epot(self, x, p=None):
        return self.compute_force(x), self.compute_epot(x)


class TestSimBenchMetrics(unittest.TestCase):
    def test_metrics(self):
        rng_dummy, rng_epot, rng_x = jax.random.split(jax.random.PRNGKey(0), 3)
        
        ethanol = ase.Atoms('C2H5OH')
        masses=ethanol.get_masses().reshape(1, -1, 1)
        atomic_numbers=ethanol.get_atomic_numbers().reshape(1, -1, 1)

        random_data = {"Epot": jax.random.normal(rng_epot, (25, 1)), "x": jax.random.normal(rng_x, (25, 9, 3))}
        data_module = InMemoryDataModule(
            train_data=random_data, 
            val_data=random_data, 
            test_data=random_data,
            static_features={
                "masses": jax.numpy.array(masses),
                "atomic_numbers": jax.numpy.array(atomic_numbers)
            })
        data_module._data = random_data  # hacky
        integration_timestep = 0.5 * ase.units.fs
        potential = DummyPotential(masses)

        mlist = [
            metrics.PlotAngularAndMeanMomentum(data_module=data_module, integration_timestep=integration_timestep),
            metrics.PlotTempAndEnergy(data_module=data_module, integration_timestep=integration_timestep, T_equilibrium=500, potential=potential),
            metrics.PlotEPotHistogram(data_module=data_module, integration_timestep=integration_timestep),
            metrics.PlotSpectrum(data_module=data_module, integration_timestep=integration_timestep),
            metrics.PlotDihedralHistogram(data_module=data_module, integration_timestep=integration_timestep, angle_index=0),
            metrics.PlotRamachandran(data_module=data_module, integration_timestep=integration_timestep),
            metrics.ForcesAreNotEnoughMetrics(data_module=data_module, integration_timestep=integration_timestep),
            metrics.LogASETraj(data_module=data_module, integration_timestep=integration_timestep),
        ]

        xs = ps = vs = fs = jax.random.normal(rng_dummy, (25, 9, 3))
        traj_data = {"xs": xs, "ps": ps, "vs": vs, "fs": fs}

        for metric in mlist:
            # test the metric with dummy data
            metric(traj_data, log=False)
