from hfm.simulation.potential import Potential
import jax.numpy as jnp


class NBodyGravityPotential(Potential):
    def __init__(self, n_balls=10, G=1.0, softening=0.1):
        self.softening = softening
        self.G = G
        self.n_balls = n_balls

        super().__init__(masses=jnp.array([1.0]).repeat(n_balls).reshape(1, -1, 1))

    def gravity_potential(self, xs: jnp.ndarray) -> jnp.ndarray:
        """Compute the gravitational potential energy."""
        dx = xs[:, :, None, :] - xs[:, None, :, :]
        r = jnp.sqrt(jnp.sum(dx**2, axis=-1) + self.softening**2)
        inv_r = jnp.where(r > 0, 1.0 / r, 0.0)

        PE_matrix = -self.G * (self.masses[:, :, None, 0] * self.masses[:, None, :, 0]) * inv_r
        return jnp.sum(jnp.triu(PE_matrix, k=1), axis=(1, 2)).reshape(-1, 1)

    def gravity_forces(self, xs: jnp.ndarray) -> jnp.ndarray:
        """Compute the gravitational forces."""
        dx = xs[:, :, None, :] - xs[:, None, :, :]
        r3 = jnp.sum(dx**2, axis=-1) + self.softening**2
        inv_r3 = jnp.where(r3 > 0, 1.0 / (r3 * jnp.sqrt(r3)), 0.0)

        F = -self.G * (dx * inv_r3[:, :, :, None]) * (self.masses[:, :, None, 0] * self.masses[:, None, :, 0])[:, :, :, None]
        return jnp.sum(F, axis=2)

    def gravity_force(self, xs: jnp.ndarray) -> jnp.ndarray:
        """Compute the force as the negative gradient of the potential."""
        # force_as_grad = jax.grad(lambda _x: -self.gravity_potential(_x).sum())(xs)
        direct_force = self.gravity_forces(xs)
        # diff = jnp.sum(jnp.abs(force_as_grad - direct_force))
        return direct_force

    def compute_force(self, x, p=None):
        return self.gravity_force(x)

    def compute_epot(self, x, p=None):
        return self.gravity_potential(x)

    def compute_force_and_epot(self, x, p=None):
        return self.compute_force(x), self.compute_epot(x)
