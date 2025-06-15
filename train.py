import os
from absl import app, flags
from ml_collections.config_flags import config_flags

import jax
from flax.training import train_state, checkpoints
import optax
from tensorboardX import SummaryWriter

from src.model import UNet
from src.data import get_cifar10_dataset
from src.training import TimeSampler, train_step
from src.sampling import sample_fn, save_image_grid

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/cifar10_config.py", "Path to configuration file.")
flags.DEFINE_string("workdir", "./checkpoints", "Work directory.")


def main(_):
    config = FLAGS.config
    workdir = FLAGS.workdir
    os.makedirs(workdir, exist_ok=True)

    # Setup logging
    writer = SummaryWriter(workdir)

    # Setup keys
    key = jax.random.PRNGKey(0)
    model_key, train_key, sample_key = jax.random.split(key, 3)

    # Load data
    dataset = get_cifar10_dataset(config.batch_size)

    # Initialize model and optimizer
    model = UNet(dim=config.model.dim, channels=config.model.channels, dim_mults=config.model.dim_mults, num_res_blocks=config.model.num_res_blocks)

    dummy_input = jnp.ones([1, config.image_size, config.image_size, config.model.channels])
    dummy_r, dummy_t = jnp.ones([1]), jnp.ones([1])
    params = model.init({"params": model_key, "dropout": model_key}, dummy_input, dummy_r, dummy_t)["params"]

    schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=config.learning_rate, warmup_steps=config.warmup_steps, decay_steps=config.num_epochs * len(dataset))
    optimizer = optax.adamw(learning_rate=schedule)

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    state = checkpoints.restore_checkpoint(workdir, state)

    # Training loop
    time_sampler = TimeSampler(config)
    step = state.step
    for epoch in range(config.num_epochs):
        for batch in dataset:
            train_key, step_key = jax.random.split(train_key)
            r, t = time_sampler.sample(step_key)

            state, loss = train_step(state, batch, r, t, train_key, config)

            if step % config.log_every_steps == 0:
                print(f"Step: {step}, Epoch: {epoch}, Loss: {loss:.4f}")
                writer.add_scalar("loss", loss, step)

            if step % config.save_every_steps == 0:
                # Save checkpoint
                checkpoints.save_checkpoint(workdir, state, step, keep=3)
                # Generate and save sample images
                sample_shape = (64, config.image_size, config.image_size, config.model.channels)
                samples = sample_fn(state, sample_key, sample_shape)
                save_image_grid(samples, os.path.join(workdir, f"sample_{step}.png"))

            step += 1


if __name__ == "__main__":
    app.run(main)
