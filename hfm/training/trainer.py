from collections import defaultdict
from functools import partial
from pathlib import Path
import jax
import jraph
from tqdm import tqdm
from hfm.datasets.utils import jraph_total_angular_momentum_3d
import wandb
import optax
import numpy as np
import jax.numpy as jnp
from flax.training import train_state

from hfm.datasets.utils import jraph_total_angular_momentum_3d
from hfm.datasets.hfm_dataset import HFMDataset
from hfm.training.ema_tracker import EMATracker
from hfm.jraph_utils import get_batch_segments, get_graph_padding_mask, get_node_padding_mask, get_number_of_graphs
from hfm.training.checkpoint import load_params_from_path, save_params_to_path
from hfm.training.ema_tracker import EMATracker


class TrainState(train_state.TrainState):
    pass


def sample_time(rng_mix, rng_u, rng_b, shape=()):
    use_unif = jax.random.bernoulli(rng_mix, p=0.02, shape=shape)
    t_u = jax.random.uniform(rng_u, shape=shape, minval=0.0, maxval=1.0)
    t_b = jax.random.beta(rng_b, a=1.0, b=2.0, shape=shape)
    return jnp.where(use_unif, t_u, t_b)


def logit_normal_sample(key, mu=-0.4, sigma=1.0, shape=()):
    # Sample from normal distribution
    z = jax.random.normal(key, shape) * sigma + mu
    # Apply logistic (sigmoid) to map to (0, 1)
    return jax.nn.sigmoid(z)


def adaptive_weighted_loss(l2_norm_sq, p=.5, c=0.001):
    weight = (l2_norm_sq + c) ** p
    return l2_norm_sq / jax.lax.stop_gradient(weight)


def rollout_bw(time_latent, graph, v_pred, f_pred):
    batch_segments = get_batch_segments(graph)
    node_mask = get_node_padding_mask(graph)

    time_latent = jnp.take(time_latent, batch_segments, axis=0)
    time_latent = jnp.where(node_mask[:, None], time_latent, 0.)

    graph_pred = jax.tree.map(lambda x: x, graph)

    graph_pred.nodes["x"] = graph.nodes["x"] - time_latent * v_pred
    graph_pred.nodes["p"] = graph.nodes["p"] - time_latent * f_pred

    return graph_pred


def node_loss_to_graph_loss(node_loss, batch_segments, num_graphs, node_mask, graph_mask):
    node_loss = jnp.where(node_mask[:, None], node_loss, 0.)
    node_loss = jraph.segment_mean(node_loss, batch_segments, num_graphs)
    node_loss = jnp.mean(node_loss, axis=-1, keepdims=True)  # average over ndims
    return jnp.where(graph_mask[:, None], node_loss, 0.)


def compute_loss(params, apply_fn, sample, rng, tmax=1., zero_t_p=1.0, large_force_threshold=3000.0, loss_weights={}, use_mass_scaling=False, use_improved_t_sampling=False):
    assert len(sample.globals["Epot"].shape) == 2, "Expecting globals to have shape (batch_size, feature_dim)"
    
    if use_improved_t_sampling:
        rng_mix, rng_u, rng_b, rng_clip, rng_dropout = jax.random.split(rng, 5)
    else:
        rng_t, rng_r, rng_clip, rng_dropout = jax.random.split(rng, 4)
    apply_individual_inputs = lambda t, g: apply_fn(params, t, g, rngs={"dropout": rng_dropout}, deterministic=False)

    batch_segments = get_batch_segments(sample)  # (num_nodes)
    num_graphs = get_number_of_graphs(sample)
    node_mask = get_node_padding_mask(sample)
    graph_mask = get_graph_padding_mask(sample)

    # Clip forces to a maximum value
    force_norm = jnp.linalg.norm(sample.nodes["f"], axis=-1, keepdims=True)  # (batch_size, num_nodes)
    
    # avoid div 0
    force_norm = jnp.where(node_mask[:, None], force_norm, 1.)
    sample.nodes["f"] = jnp.where(((force_norm < large_force_threshold) | ~node_mask[:, None]), sample.nodes["f"], large_force_threshold * (sample.nodes["f"] / force_norm))

    if use_improved_t_sampling:
        t = sample_time(rng_mix, rng_u, rng_b, shape=(num_graphs,))
        t = t.reshape(-1, 1) * tmax
        zero_t = jax.random.uniform(rng_clip, shape=(t.shape[0], 1))
        t = jnp.where(zero_t < zero_t_p, jnp.zeros_like(t), t)  # Clip t to zero with zero_t_p probability
    else:
        t = logit_normal_sample(rng_t, shape=(num_graphs,))
        r = logit_normal_sample(rng_r, shape=(num_graphs,))

        t = jnp.abs(t - r).reshape(-1, 1) * tmax
        zero_t = jax.random.uniform(rng_clip, shape=(t.shape[0], 1))
        t = jnp.where(zero_t < zero_t_p, jnp.zeros_like(t), t)  # Clip t to zero with zero_t_p probability

    def zero_tangent_like(tree):
        def make_zero(x):
            if jnp.issubdtype(x.dtype, jnp.floating):
                return jnp.zeros_like(x)
            # non-float leaves (ints, bools, etc.) → special 'no tangent' object
            return jnp.zeros_like(x, dtype=jax.float0)
        return jax.tree.map(make_zero, tree)

    tangent_graph = zero_tangent_like(sample)
    tangent_graph.nodes["x"] = sample.nodes["v"]
    tangent_graph.nodes["p"] = sample.nodes["f"]

    (mean_v_pred, mean_f_pred, energy_pred), (mean_v_jvp, mean_f_jvp, _) = jax.jvp(apply_individual_inputs, (t, sample), (-jnp.ones_like(t), tangent_graph))

    # Mean flow loss targets
    time_atom = jnp.take(t, batch_segments)
    time_atom = jnp.where(node_mask, time_atom, 0.)[:, None]

    mean_v_target = sample.nodes["v"] + time_atom * mean_v_jvp
    mean_v_target = jax.lax.stop_gradient(mean_v_target)

    mean_f_target = sample.nodes["f"] + time_atom * mean_f_jvp
    mean_f_target = jax.lax.stop_gradient(mean_f_target)
    
    # Mean Force loss per batch
    mean_f_loss = node_loss_to_graph_loss((mean_f_pred - mean_f_target) ** 2, batch_segments, num_graphs, node_mask, graph_mask)

    mean_v_loss_per_node = (mean_v_pred - mean_v_target) ** 2
    # Mean Velocity loss per batch
    if use_mass_scaling:
        mean_v_loss_per_node = sample.nodes["masses"] * mean_v_loss_per_node

    mean_v_loss = node_loss_to_graph_loss(mean_v_loss_per_node, batch_segments, num_graphs, node_mask, graph_mask)

    # Energy loss per batch
    energy_loss = jnp.mean((sample.globals["Epot"] - energy_pred) ** 2, axis=-1, keepdims=True)
    energy_loss = jnp.where(graph_mask[:, None], energy_loss, 0.)
    
    # MAE Energy loss
    energy_mae = jnp.mean(jnp.abs(sample.globals["Epot"] - energy_pred), axis=-1, keepdims=True)
    energy_mae = jnp.where(graph_mask[:, None], energy_mae, 0.)

    # MAE Energy loss
    energy_mae = jnp.mean(jnp.abs(sample.globals["Epot"] - energy_pred), axis=-1, keepdims=True)
    energy_mae = jnp.where(graph_mask[:, None], energy_mae, 0.)

    # L2 loss per batch (for logging)
    l2_loss = mean_f_loss * loss_weights["mean_force"] + mean_v_loss * loss_weights["mean_velocity"] + energy_loss * loss_weights["energy"]

    # Adaptive weighted loss
    mean_f_loss_adaptive = adaptive_weighted_loss(mean_f_loss) * loss_weights["mean_force"]
    mean_v_loss_adaptive = adaptive_weighted_loss(mean_v_loss) * loss_weights["mean_velocity"]
    energy_loss_adaptive = adaptive_weighted_loss(energy_loss) * loss_weights["energy"]
    
    l2_loss_adaptive = mean_f_loss_adaptive + mean_v_loss_adaptive + energy_loss_adaptive

    # L2 loss for velocity and force predictions
    # regular L2 loss
    v_loss = node_loss_to_graph_loss(jnp.abs(mean_v_pred - sample.nodes["v"]), batch_segments, num_graphs, node_mask, graph_mask)
    f_loss = node_loss_to_graph_loss(jnp.abs(mean_f_pred - sample.nodes["f"]), batch_segments, num_graphs, node_mask, graph_mask)

    # mask out predictions where t > 0
    # to get a good estimate for how accurate
    # the model predicts forces and velocities for t=0
    mask_t = jnp.where(t > 0, 1.0, 0.0)
    mask_t = jnp.where(graph_mask[:, None], mask_t, 0.)
    v_loss_t = jnp.sum(v_loss * mask_t) / jnp.sum(jnp.ones_like(v_loss) * mask_t)
    f_loss_t = jnp.sum(f_loss * mask_t) / jnp.sum(jnp.ones_like(f_loss) * mask_t)

    mask = jnp.where(t > 0, 0.0, 1.0)
    mask = jnp.where(graph_mask[:, None], mask, 0.)
    v_loss = jnp.sum(v_loss * mask) / jnp.sum(jnp.ones_like(v_loss) * mask)
    f_loss = jnp.sum(f_loss * mask) / jnp.sum(jnp.ones_like(f_loss) * mask)

    aux = {}
    aux.update({
        'L2 train loss': jnp.mean(l2_loss),
        'velocity loss (t=0) MAE': v_loss,
        'velocity loss (t>0) MAE': v_loss_t,
        'force loss (t=0) MAE': f_loss,
        'force loss (t>0) MAE': f_loss_t,
        'energy loss MAE': jnp.mean(energy_mae),
        'mean velocity loss (L2)': jnp.mean(mean_v_loss),
        'mean force loss (L2)': jnp.mean(mean_f_loss),
        'energy loss (L2)': jnp.mean(energy_loss),
        'mean velocity loss (adaptive)': jnp.mean(mean_v_loss_adaptive),
        'mean force loss (adaptive)': jnp.mean(mean_f_loss_adaptive),
        'energy loss (adaptive)': jnp.mean(energy_loss_adaptive),
    })

    return jnp.mean(l2_loss_adaptive), aux


# Training step
@partial(jax.jit, static_argnames=["validation", "large_force_threshold", "use_mass_scaling", "use_improved_t_sampling"])
def train_step(validation, large_force_threshold, use_mass_scaling, use_improved_t_sampling, state, sample, rng, tmax, loss_weights, zero_t_p=1.0):
    loss_fn = lambda params: compute_loss(params=params, 
                                          apply_fn=state.apply_fn, 
                                          sample=sample, 
                                          rng=rng, 
                                          tmax=tmax, 
                                          zero_t_p=zero_t_p, 
                                          loss_weights=loss_weights, 
                                          large_force_threshold=large_force_threshold,
                                          use_mass_scaling=use_mass_scaling,
                                          use_improved_t_sampling=use_improved_t_sampling)

    if validation:
        loss, aux = loss_fn(state.params)
        return loss, aux
    else:
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, aux, grads


class HFMTrainer:
    def __init__(self, 
                 data_module,
                 temperature_mean,
                 temperature_std,
                 n_dim,
                 rotation_augmentation,
                 zero_momenta_p, 
                 zero_t_p, 
                 t_max, 
                 model,
                 n_epochs=2_000, 
                 validate_after_n_epochs=10,
                 batch_size=512,
                 optimizer_parameters=None,
                 scale_parameters=None,
                 workdir="", 
                 checkpoint_name="",
                 save_params_every=200,
                 large_force_threshold=3000.0,
                 debug_nans=False,
                 use_mass_scaling=False,
                 use_improved_t_sampling=False,
                 reset_to_ema_every_epoch=False,
                 ema_beta=0.99,
                 load_momenta_from_force_dataset=False,
                 zero_rot_p=0.0,
                 zero_drift=True):

        if debug_nans:
            jax.config.update("jax_debug_nans", True)
            jax.config.update("jax_debug_infs", True)

        self.zero_rot_p = zero_rot_p
        self.zero_drift = zero_drift
        self.temperature_mean = temperature_mean
        self.temperature_std = temperature_std
        self.n_dim = n_dim
        self.rotation_augmentation = rotation_augmentation
        self.zero_momenta_p = zero_momenta_p
        self.zero_t_p = zero_t_p
        self.t_max = t_max
        self.model = model
        self.n_epochs = n_epochs
        self.validate_after_n_epochs = validate_after_n_epochs
        self.batch_size = batch_size
        self.optimizer_parameters = optimizer_parameters
        self.scale_parameters = scale_parameters
        self.workdir = workdir
        self.checkpoint_name = checkpoint_name
        self.save_params_every = save_params_every
        self.large_force_threshold = large_force_threshold
        self.reset_to_ema_every_epoch = reset_to_ema_every_epoch
        self.ema_beta = ema_beta  # ema beta, nequip uses 0.99, mean flow uses 0.9999
        self.load_momenta_from_force_dataset = load_momenta_from_force_dataset
        self.use_mass_scaling = use_mass_scaling
        self.use_improved_t_sampling = use_improved_t_sampling
        self.data_module = data_module
        self.global_norm_fn = jax.jit(optax.global_norm)

        self.train_dataset = self.wrap_hfm_dataset(data_module.train_dataset)
        self.val_dataset = self.wrap_hfm_dataset(data_module.val_dataset)

        if self.optimizer_parameters is None:
            self.optimizer_parameters = {
                "init_value": 1e-6,
                "peak_value": 1e-4,
                "end_value": 1e-8,
                "warmup_pct": 0.01,
                "adam_b1": 0.9,
                "adam_b2": 0.95,
                "adam_weight_decay": 0.0,
                "max_norm": 5.0,
            }

        if self.scale_parameters is None:
            self.scale_parameters = {
                "mean_force": 1.0,
                "mean_velocity": 1.0,
                "energy": 1.0,
                "energy_conservation": 1.0,
            }

    def wrap_hfm_dataset(self, dataset):
        return HFMDataset(
            dataset,
            temperature_mean = self.temperature_mean, 
            temperature_std = self.temperature_std, 
            n_dim = self.n_dim, 
            rotation_augmentation = self.rotation_augmentation, 
            zero_momenta_p = self.zero_momenta_p,
            load_momenta_from_force_dataset = self.load_momenta_from_force_dataset,
            zero_rot_p = self.zero_rot_p,
            zero_drift = self.zero_drift,
        )
    
    def shutdown(self):
        self.data_module.shutdown()

    def make_opt(self):
        n_steps = int(self.n_epochs * len(self.train_dataset) / self.batch_size)

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=self.optimizer_parameters["init_value"],
            peak_value=self.optimizer_parameters["peak_value"],
            warmup_steps=int(self.optimizer_parameters["warmup_pct"] * n_steps),
            decay_steps=int((1 - self.optimizer_parameters["warmup_pct"]) * n_steps),
            end_value=self.optimizer_parameters["end_value"],
        )

        opt = optax.adamw(
            lr_schedule, 
            b1=self.optimizer_parameters["adam_b1"],
            b2=self.optimizer_parameters["adam_b2"], 
            weight_decay=self.optimizer_parameters["adam_weight_decay"],
        )

        chained_opt = optax.chain(
            optax.clip_by_global_norm(self.optimizer_parameters["max_norm"]),
            optax.zero_nans(),
            opt
        )

        return chained_opt, lr_schedule
    
    def model_init(self, rng_init):
        rng_init, rng_sample = jax.random.split(rng_init, 2)
        sample = self.train_dataset.get_example_batch(rng_sample, self.batch_size)
        return self.model.init(rng_init, jnp.ones((sample.nodes["x"].shape[0], 1)), sample)
    
    def __call__(self, seed=42, debug=False, load_params=False, verbose=True):
        workdir = Path(self.workdir)
        rng = jax.random.PRNGKey(seed)

        if load_params:
            print(f"Load params from {load_params}...")
            params = load_params_from_path(load_params, model_version=self.model.model_version)
        else:
            print("Initializing model...")
            rng, rng_init = jax.random.split(rng, 2)
            params = self.model_init(rng_init)

        if self.n_epochs <= 0:
            print("No training epochs specified, skipping training.")
            return params
        param_count = sum(
            x.size for x in jax.tree_util.tree_leaves(params)
        )
        print("Number of params:", param_count)

        tx, lr_schedule = self.make_opt()
        state = TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

        ema = EMATracker(beta=self.ema_beta)
        ema.initialize(params)
        
        print("Starting training...")
        first_sample = None

        for epoch in range(self.n_epochs):
            rng, rng_epoch = jax.random.split(rng, 2)
            losses = []

            for iteration, sample in tqdm(enumerate(self.train_dataset.next_epoch(rng_epoch, self.batch_size)), total=len(self.train_dataset) // self.batch_size, leave=False):
                if debug: 
                    if first_sample is None:
                        first_sample = sample
                    sample = first_sample

                rng, rng_train = jax.random.split(rng, 2)
                state, loss, aux, grads = train_step(validation=False,
                                              large_force_threshold=self.large_force_threshold,
                                              use_mass_scaling=self.use_mass_scaling,
                                              use_improved_t_sampling=self.use_improved_t_sampling,
                                              state=state, 
                                              sample=sample, 
                                              rng=rng_train, 
                                              tmax=self.t_max, 
                                              loss_weights=self.scale_parameters, 
                                              zero_t_p=self.zero_t_p)
                ema.update(state.params)
                losses.append(loss)

                if iteration % 100 == 0 and wandb.run is not None:
                    logs = {"train/loss": loss.item()}
                    logs.update({"train/" + k: v.item() for k, v in aux.items()})
                    logs["lr"] = lr_schedule(state.step)
                    logs["epoch"] = epoch
                    logs["tmax"] = self.t_max
                    logs["global_step"] = state.step
                    logs["gradient_norm"] = self.global_norm_fn(grads)

                    if debug:
                        print("log wandb", logs)
                        
                    wandb.log(logs)

                if debug and iteration == 5000 and epoch == 0:
                    print("Debugging mode: saving checkpoint after 5000 iterations")
                    path = str(workdir / f"{self.checkpoint_name}params_debug.pkl")
                    save_params_to_path(state.params, path, model_version=self.model.model_version)

            if verbose:
                print(f"Epoch {epoch}, Loss: {np.array(losses).mean():.4f}")

            # save after each n epochs, save after 5 to do a quick sanity check
            if epoch % self.save_params_every == 0 or epoch == 5:
                # save intermediate checkpoint
                path = str(workdir / f"{self.checkpoint_name}params_epoch{epoch}.pkl")
                save_params_to_path(state.params, path, model_version=self.model.model_version)

                # save intermediate EMA checkpoint
                path = str(workdir / f"{self.checkpoint_name}paramsEMA_epoch{epoch}.pkl")
                save_params_to_path(ema.shadow_params, path, model_version=self.model.model_version)

            # validatate after each n epochs
            # AND after the last train epoch
            if (epoch % self.validate_after_n_epochs == 0) or (epoch == self.n_epochs - 1):
                rng, rng_epoch = jax.random.split(rng, 2)
                losses = defaultdict(list)

                for iteration, sample in tqdm(enumerate(self.val_dataset.next_epoch(rng_epoch, self.batch_size)), total=len(self.val_dataset) // self.batch_size, leave=False):
                    rng, rng_val = jax.random.split(rng, 2)
                    loss, aux = train_step(validation=True,
                                            large_force_threshold=self.large_force_threshold,
                                            use_mass_scaling=self.use_mass_scaling,
                                            use_improved_t_sampling=self.use_improved_t_sampling,
                                            state=state, 
                                            sample=sample, 
                                            rng=rng_val, 
                                            tmax=self.t_max, 
                                            loss_weights=self.scale_parameters, 
                                            zero_t_p=self.zero_t_p)
                    
                    losses["val/loss"].append(loss.item())
                    for k, v in aux.items():
                        losses["val/" + k].append(v.item())

                logs = {k: np.array(v).mean() for k, v in losses.items()}
                if wandb.run is not None:
                    # Log mean validation losses
                    wandb.log(logs)
                
                print(f"Epoch {epoch}, Validation summary: {logs}")

            # at the end of each epoch, reset params to EMA
            if self.reset_to_ema_every_epoch:
                state = state.replace(params=jax.tree.map(lambda x: x, ema.shadow_params))

        # save final checkpoint
        path = str(workdir / f"{self.checkpoint_name}params_last.pkl")
        save_params_to_path(state.params, path, model_version=self.model.model_version)

        # save final EMA checkpoint
        path = str(workdir / f"{self.checkpoint_name}paramsEMA_last.pkl")
        save_params_to_path(ema.shadow_params, path, model_version=self.model.model_version)

        return state.params
