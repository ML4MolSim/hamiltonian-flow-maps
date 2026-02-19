from hfm.simulation.potential import Potential
import jax.numpy as jnp


class DummyPotential(Potential):
    def __init__(self, masses):
        super().__init__(masses=masses)

    def compute_force(self, x, p):
        return jnp.zeros(x.shape)

    def compute_epot(self, x, p):
        return jnp.zeros((x.shape[0], 1))

    def compute_force_and_epot(self, x, p):
        return self.compute_force(x, p), self.compute_epot(x, p)
