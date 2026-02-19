from abc import ABC, abstractmethod
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


class IntegrationFilter(ABC):
    """Integration filter
    An auxiliary variable (dictionary) is passed for each call (filter_aux) which will be shared among all filters.
    It will also be passed to the out call such that filters can store auxiliary variables.
    """

    def init_aux(self, x0, p0, masses, filter_aux: dict):
        return x0, p0, filter_aux

    @abstractmethod
    def in_call(self, x, p, integration_timestep, masses, filter_aux: dict, rng):
        """Filters are expected to return x,p,filter_aux"""
        pass

    @abstractmethod
    def out_call(self, x, p, integration_timestep, v, f, masses, filter_aux: dict, rng):
        """Filters are expected to return x,p,filter_aux"""
        pass


class Integrator(ABC):
    def __init__(self, integration_timestep=None, masses=None, filters=None, nested_integrator=None, multistep_nested=False):
        assert masses is not None, "masses must be provided to the integrator"

        # multi step nested allows for multiple steps in nested integrators
        # if multistep_nested = False, it means that nested integrators only do a single step
        self.multistep_nested = multistep_nested

        self.masses = masses
        self._integration_timestep = integration_timestep # only for convenience (simbench config) and bw compatibility
        # use integration_timestep in __call__ instead when calling directly
        self.filters = [] if filters is None else filters
        self.nested_integrator = nested_integrator

    def add_integration_filter(self, filter: IntegrationFilter):
        # filters will be added from innermost to outermost
        self.filters.append(filter)

    def init_aux(self, x, p):
        aux = {}
        if self.nested_integrator is not None:
            aux = self.nested_integrator.init_aux(x, p)

        return aux

    def init_filter_aux(self, x, p, masses):
        filter_aux = {}
        for filter in reversed(self.filters):
            x, p, filter_aux = filter.init_aux(x, p, masses, filter_aux)

        if self.nested_integrator is not None:
            filter_aux = self.nested_integrator.init_filter_aux(x, p, masses)

        return filter_aux
    
    @abstractmethod
    def integration_step(self, x, p, integration_timestep, aux, filter_aux, rng):
        """Perform a single integration step."""
        pass

    @partial(jax.jit, static_argnums=(0))
    def integrate_with_filters(self, x, p, integration_timestep, aux, filter_aux, rng):
        return self._integrate_with_filters(x, p, integration_timestep, aux, filter_aux, rng)

    @partial(jax.jit, static_argnums=(0, 3))
    def integrate_with_filters_multiple_nested(self, x, p, integration_timestep, aux, filter_aux, rng):
        return self._integrate_with_filters(x, p, integration_timestep, aux, filter_aux, rng)

    def _integrate_with_filters(self, x, p, integration_timestep, aux, filter_aux, rng):
        # Create a copy to make sure we do not in place update the filters which can have nasty side effects.
        # Not needed as long as this function is jitted but just to be sure.
        filter_aux = jax.tree_util.tree_map(lambda x: x, filter_aux)

        rng_integrate, rng_filter_in, rng_filter_out = jax.random.split(rng, 3)
        rng_filter_in = jax.random.split(rng_filter_in, len(self.filters))
        rng_filter_out = jax.random.split(rng_filter_out, len(self.filters))

        # start with outermost filter
        for filter, rng in zip(reversed(self.filters), rng_filter_in):
            x, p, filter_aux = filter.in_call(x, p, integration_timestep, self.masses, filter_aux, rng)

        x, p, v, f, aux, filter_aux = self.integration_step(x, p, integration_timestep, aux, filter_aux, rng_integrate)

        for filter, rng in zip(self.filters, rng_filter_out):
            x, p, filter_aux = filter.out_call(x, p, integration_timestep, v, f, self.masses, filter_aux, rng)

        return x, p, v, f, aux, filter_aux

    @partial(jax.jit, static_argnums=(0, 1, 6))
    def do_steps_multiple_nested(self, unroll, x, p, aux, filter_aux, integration_timestep, rngs):
        f_integrate = self.integrate_with_filters_multiple_nested
        return self._do_steps(unroll, x, p, aux, filter_aux, integration_timestep, rngs, f_integrate)

    @partial(jax.jit, static_argnums=(0, 1))
    def do_steps(self, unroll, x, p, aux, filter_aux, integration_timestep, rngs):
        f_integrate = self.integrate_with_filters
        return self._do_steps(unroll, x, p, aux, filter_aux, integration_timestep, rngs, f_integrate)
    
    def _do_steps(self, unroll, x, p, aux, filter_aux, integration_timestep, rngs, f_integrate):
        def scan_step(carry, rng):
            x, p, aux, filter_aux = carry
            x, p, v, f, aux, filter_aux = f_integrate(
                x, p, integration_timestep, aux, filter_aux, rng
            )
            return (x, p, aux, filter_aux), (x, p, v, f)

        (x, p, aux, filter_aux), (cur_xs, cur_ps, cur_vs, cur_fs) = (
            jax.lax.scan(
                scan_step, (x, p, aux, filter_aux), rngs, unroll=unroll
            )
        )

        return (x, p, aux, filter_aux), (cur_xs, cur_ps, cur_vs, cur_fs)

    def call_nested_integrator(self, 
        x, 
        p,         
        integration_time,
        rng,
        aux,
        filter_aux
    ):
        xs, ps, vs, fs, aux, filter_aux = self.nested_integrator._nested_call(x, p, integration_time, rng, aux=aux, filter_aux=filter_aux)
        return xs, ps, vs, fs, aux, filter_aux
    
    def _nested_call(
        self,
        x,
        p,
        integration_time,
        rng,
        aux,
        filter_aux,
        save_every=1,
        integration_timestep=None,
    ):
        assert x.ndim == 3, "x and p must be batched"
        assert x.shape == p.shape

        if integration_timestep is None:
            integration_timestep = self._integration_timestep

        # Here we disable multiple steps for nested integrators!!
        # We can re-enable this but then we need to re-compile if integration_timestep changes
        assert filter_aux is not None, "filter_aux must be provided for nested integrators"
        assert aux is not None, "aux must be provided for nested integrators"

        if self.multistep_nested:  # everything needs to be jax arrays and on GPU
            if self.multistep_nested:
                assert (
                    integration_timestep is not None
                ), "integration_timestep must be provided if not single-step nested"

            n_steps = int(integration_time / integration_timestep)
            assert save_every <= n_steps, "save_every must be less than or equal to n_steps"

            xs, ps, vs, fs = [], [], [], []
            for i, rng in enumerate(jax.random.split(rng, n_steps)):
                x, p, v, f, aux, filter_aux = self.integrate_with_filters_multiple_nested(
                    x, p, integration_timestep, aux, filter_aux, rng
                )

                if (i + 1) % save_every == 0:
                    xs.append(x)
                    ps.append(p)
                    vs.append(v)
                    fs.append(f)
        else:
            if integration_timestep is not None:
                print("WARNING: Specifying integration_timestep for nested integrators is deprecated and will be overridden.")

            xs, ps, vs, fs = [], [], [], []
            x, p, v, f, aux, filter_aux = self.integrate_with_filters(
                x, p, integration_time, aux, filter_aux, rng
            ) 

            xs.append(x)
            ps.append(p)
            vs.append(v)
            fs.append(f)

        xs = jnp.stack(xs, axis=1)
        ps = jnp.stack(ps, axis=1)
        vs = jnp.stack(vs, axis=1)
        fs = jnp.stack(fs, axis=1)

        return xs, ps, vs, fs, aux, filter_aux

    def __call__(
        self,
        x,
        p,
        integration_time,
        rng,
        save_every=1,
        intermediate_steps=1000,
        unroll=1,
        squeeze_batchdim=False,
        integration_timestep=None,
        filter_aux = None,
        aux = None
    ):
        assert x.ndim == 3, "x and p must be batched"
        assert x.shape == p.shape
        assert save_every <= intermediate_steps

        f_do_steps = self.do_steps
        if self.nested_integrator is not None and self.nested_integrator.multistep_nested:
            f_do_steps = self.do_steps_multiple_nested

        if integration_timestep is None:
            integration_timestep = self._integration_timestep

        assert (
            integration_timestep is not None
        ), "integration_timestep must be provided if not nested"

        n_simulations = x.shape[0]
        sample_dim = x.shape[1:]
        aux = self.init_aux(x, p)  # auxiliary integration variables, if any
        filter_aux = self.init_filter_aux(x, p, self.masses)
        n_steps = int(integration_time / integration_timestep)
        assert save_every <= n_steps, "save_every must be less than or equal to n_steps"

        # we pad to a multiple of intermediate_steps, and then just truncate to n_total_steps
        # this operation does a ceiling division
        n_total_steps = -(n_steps // -intermediate_steps) * intermediate_steps

        rngs = jax.random.split(rng, n_total_steps)

        # pre-allocate cpu memory
        xs, ps, vs, fs = (
            np.zeros((n_simulations, n_steps // save_every, *sample_dim)),
            np.zeros((n_simulations, n_steps // save_every, *sample_dim)),
            np.zeros((n_simulations, n_steps // save_every, *sample_dim)),
            np.zeros((n_simulations, n_steps // save_every, *sample_dim)),
        )

        with tqdm(total=n_steps, leave=False) as pbar:
            for start_idx in range(0, n_total_steps, intermediate_steps):
                end_idx = min(start_idx + intermediate_steps, n_steps)
                end_idx_total_steps = min(
                    start_idx + intermediate_steps, n_total_steps
                )

                # do <intermediate steps> on GPU and then move the data to CPU
                (x, p, aux, filter_aux), (cur_xs, cur_ps, cur_vs, cur_fs) = (
                    f_do_steps(
                        unroll, x, p, aux, filter_aux, integration_timestep, rngs[start_idx:end_idx_total_steps]
                    )
                )

                simulated_indices = (
                    np.arange(start_idx, end_idx_total_steps) + 1
                )  # +1 because we consider the initial state
                save_indices = (simulated_indices % save_every == 0) & (
                    simulated_indices <= n_steps
                )

                # this consideres that we don't store every step
                adj_start_idx = start_idx // save_every
                adj_end_idx = adj_start_idx + np.sum(save_indices)

                xs[:, adj_start_idx:adj_end_idx] = np.swapaxes(
                    np.array(cur_xs)[save_indices], 0, 1
                )
                ps[:, adj_start_idx:adj_end_idx] = np.swapaxes(
                    np.array(cur_ps)[save_indices], 0, 1
                )
                vs[:, adj_start_idx:adj_end_idx] = np.swapaxes(
                    np.array(cur_vs)[save_indices], 0, 1
                )
                fs[:, adj_start_idx:adj_end_idx] = np.swapaxes(
                    np.array(cur_fs)[save_indices], 0, 1
                )

                pbar.update(end_idx - start_idx)

        if squeeze_batchdim and n_simulations == 1:
            xs = xs[0]
            ps = ps[0]
            vs = vs[0]
            fs = fs[0]

        return xs, ps, vs, fs
