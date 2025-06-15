import jax
import jax.numpy as jnp
from jax.lax import stop_gradient


class TimeSampler:
    """Samples time steps r and t for training."""

    def __init__(self, config):
        self.config = config.meanflow
        self.batch_size = config.batch_size

    def sample(self, key):
        key_r, key_t = jax.random.split(key)

        if self.config.rt_dist == "uniform":
            r = jax.random.uniform(key_r, (self.batch_size,))
            t = jax.random.uniform(key_t, (self.batch_size,))
        elif self.config.rt_dist == "lognorm":
            mu = self.config.rt_sampler_params["mu"]
            sigma = self.config.rt_sampler_params["sigma"]
            r_norm = jax.random.normal(key_r, (self.batch_size,))
            t_norm = jax.random.normal(key_t, (self.batch_size,))
            r = jax.nn.sigmoid(mu + sigma * r_norm)
            t = jax.nn.sigmoid(mu + sigma * t_norm)
        else:
            raise ValueError(f"Unknown time distribution: {self.config.rt_dist}")

        # Ensure r < t
        r, t = jnp.minimum(r, t), jnp.maximum(r, t)

        # Set a portion of samples to have r = t
        mask_key, _ = jax.random.split(key, 2)
        mask = jax.random.uniform(mask_key, (self.batch_size,)) > self.config.r_is_not_t_ratio
        r = jnp.where(mask, t, r)

        return r, t


def meanflow_loss_fn(params, model_apply_fn, x_batch, r, t, key, config):
    """Calculates the MeanFlow loss."""
    noise_key, dropout_key = jax.random.split(key)
    e = jax.random.normal(noise_key, x_batch.shape)

    t_b = t.reshape(-1, 1, 1, 1)
    r_b = r.reshape(-1, 1, 1, 1)

    z = (1 - t_b) * x_batch + t_b * e
    v = e - x_batch

    def model_forward(primals):
        _z, _r, _t = primals
        return model_apply_fn({"params": params}, _z, _r, _t, rngs={"dropout": dropout_key})

    primals = (z, r, t)
    tangents = (v, jnp.zeros_like(r), jnp.ones_like(t))

    u_pred, dudt = jax.jvp(model_forward, (primals,), (tangents,))
    u_tgt = v - (t_b - r_b) * dudt
    error = u_pred - stop_gradient(u_tgt)

    # Adaptive loss weighting from the paper
    error_norm = jnp.sqrt(jnp.mean(error**2, axis=(1, 2, 3), keepdims=True))
    weight = stop_gradient(1.0 / ((error_norm**2 + 1e-3) ** config.meanflow.loss_p))
    loss = jnp.mean(weight * (error**2))

    return loss


@jax.jit
def train_step(state, batch, r, t, key, config):
    """Performs a single training step."""
    loss, grads = jax.value_and_grad(meanflow_loss_fn)(state.params, state.apply_fn, batch, r, t, key, config)
    state = state.apply_gradients(grads=grads)
    return state, loss
