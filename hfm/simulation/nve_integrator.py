from abc import ABC
from hfm.simulation.base import Integrator


class NVEIntegrator(Integrator, ABC):
    def __init__(self, 
                 potential, 
                 integration_timestep=None, 
                 **kwargs):

        self.potential = potential
        super().__init__(integration_timestep, potential.masses, **kwargs)


class EulerIntegrator(NVEIntegrator):
    def integration_step(self, x, p, integration_timestep, aux, filter_aux, rng):
        v = self.potential.compute_velocity(x, p)
        f = self.potential.compute_force(x, p)
        x += integration_timestep * v
        p += integration_timestep * f
        return x, p, v, f, aux, filter_aux
    

class ImplicitEulerIntegrator(NVEIntegrator):
    def integration_step(self, x, p, integration_timestep, aux, filter_aux, rng):
        f = self.potential.compute_force(x, p)
        p += integration_timestep * f
        v = self.potential.compute_velocity(x, p)
        x += integration_timestep * v
        return x, p, v, f, aux, filter_aux
    

class VelocityVerletIntegrator(NVEIntegrator):
    def integration_step(self, x, p, integration_timestep, aux, filter_aux, rng):
        f = self.potential.compute_force(x, p)
        p_half = p + 0.5 * integration_timestep * f

        v = self.potential.compute_velocity(x, p_half)
        x = x + integration_timestep * v

        f = self.potential.compute_force(x, p_half)
        p = p_half + 0.5 * integration_timestep * f

        return x, p, v, f, aux, filter_aux
