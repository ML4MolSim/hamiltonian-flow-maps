from hfm.simulation.potential import Potential
import jax.numpy as jnp
import jax


class HarmonicPotential(Potential):
    def __init__(self, k: float = 1.0, mass: float = 1.0):
        super().__init__(masses=jnp.array([mass]).reshape(1, -1, 1))
        self.k = k

    def compute_epot(self, x, p=None) -> jnp.ndarray:
        """U = 1/2 * k * (x^2 + y^2)"""
        return 0.5 * self.k * jnp.sum(x**2, axis=-1)

    def compute_force(self, x, p=None) -> jnp.ndarray:
        """F = -grad U = -k * x"""
        return -self.k * x
    
    def compute_force_and_epot(self, x, p=None):
        return self.compute_force(x), self.compute_epot(x)


class AnharmonicPotential(Potential):
    def __init__(self, k: float = 1.0, alpha: float = 0.5, mass: float = 1.0):
        super().__init__(masses=jnp.array([mass]).reshape(1, -1, 1))
        self.k = k
        self.alpha = alpha

    def compute_epot(self, x, p=None) -> jnp.ndarray:
        """U = 1/2 * k * r^2 + 1/4 * alpha * r^4"""
        r2 = jnp.sum(x**2, axis=-1)
        return 0.5 * self.k * r2 + 0.25 * self.alpha * r2**2

    def compute_force(self, x, p=None) -> jnp.ndarray:
        """F = -grad U = -k * x - alpha * r^2 * x"""
        r2 = jnp.sum(x**2, axis=-1, keepdims=True)
        return -self.k * x - self.alpha * r2 * x

    def compute_force_and_epot(self, x, p=None):
        return self.compute_force(x), self.compute_epot(x)
    

class HenonHeilesPotential(Potential):
    def __init__(self, lmbda: float = 1.0, mass: float = 1.0):
        super().__init__(masses=jnp.array([mass]).reshape(1, -1, 1))
        self.lmbda = lmbda
        self.mass = mass

    def compute_epot(self, x, p=None) -> jnp.ndarray:
        """
        V = 1/2(x^2 + y^2) + lambda(x^2*y - 1/3*y^3)
        x input shape: (..., 2)
        """
        x_coord = x[..., 0]
        y_coord = x[..., 1]
        
        harmonic = 0.5 * (x_coord**2 + y_coord**2)
        coupling = self.lmbda * (x_coord**2 * y_coord - (1.0/3.0) * y_coord**3)
        return harmonic + coupling

    def compute_force(self, x, p=None) -> jnp.ndarray:
        # F = -grad(V)
        return -jax.grad(lambda pos: jnp.sum(self.compute_epot(pos)))(x)

    def compute_force_and_epot(self, x, p=None):
        return self.compute_force(x), self.compute_epot(x)
    

class SpringPendulumPotential(Potential):
    def __init__(self, k: float = 10.0, L0: float = 1.0, g: float = 9.81, mass: float = 1.0):
        super().__init__(masses=jnp.array([mass]).reshape(1, -1, 1))
        self.k = k
        self.L0 = L0
        self.g = g
        self.mass = mass

    def compute_epot(self, x, p=None) -> jnp.ndarray:
        """
        V = mgy + 1/2 * k * (r - L0)^2
        where r = sqrt(x^2 + y^2)
        """
        x_coord = x[..., 0]
        y_coord = x[..., 1]
        
        r = jnp.sqrt(x_coord**2 + y_coord**2)
        
        v_grav = self.mass * self.g * y_coord
        v_spring = 0.5 * self.k * (r - self.L0)**2
        
        return v_grav + v_spring

    def compute_force(self, x, p=None) -> jnp.ndarray:
        return -jax.grad(lambda pos: jnp.sum(self.compute_epot(pos)))(x)

    def compute_force_and_epot(self, x, p=None):
        return self.compute_force(x), self.compute_epot(x)


class BarbanisPotential(Potential):
    def __init__(self, lmbda: float = 1.0, mass: float = 1.0, omega_x: float = 1.0, omega_y: float = 1.0):
        super().__init__(masses=jnp.array([mass]).reshape(1, -1, 1))
        self.lmbda = lmbda
        self.mass = mass
        self.omega_x = omega_x
        self.omega_y = omega_y

    def compute_epot(self, x, p=None) -> jnp.ndarray:
        """
        V = 1/2 * (omega_x^2 * x^2 + omega_y^2 * y^2) + lambda * x^2 * y^2
        This potential is 'closed' and will keep the particle bounded.
        """
        x_coord = x[..., 0]
        y_coord = x[..., 1]
        
        harmonic = 0.5 * (self.omega_x**2 * x_coord**2 + self.omega_y**2 * y_coord**2)
        coupling = self.lmbda * (x_coord**2 * y_coord**2)
        
        return harmonic + coupling

    def compute_force(self, x, p=None) -> jnp.ndarray:
        # F = -grad(V)
        # Using jnp.sum to handle batched inputs correctly for the scalar-output grad function
        return -jax.grad(lambda pos: jnp.sum(self.compute_epot(pos)))(x)

    def compute_force_and_epot(self, x, p=None):
        return self.compute_force(x), self.compute_epot(x)
