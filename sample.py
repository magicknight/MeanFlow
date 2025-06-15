import os
from absl import app, flags
from ml_collections.config_flags import config_flags

import jax
import jax.numpy as jnp
from flax.training import checkpoints

from src.model import UNet
from src.sampling import sample_fn, save_image_grid

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/cifar10_config.py", "Path to configuration file.")
flags.DEFINE_string("checkpoint_dir", "./checkpoints", "Directory with checkpoints.")
flags.DEFINE_string("output_path", "./samples/generated_grid.png", "Path to save the generated image grid.")
flags.DEFINE_integer("num_samples", 64, "Number of samples to generate.")


def main(_):
    config = FLAGS.config
    os.makedirs(os.path.dirname(FLAGS.output_path), exist_ok=True)

    # Initialize model
    model = UNet(dim=config.model.dim, channels=config.model.channels, dim_mults=config.model.dim_mults, num_res_blocks=config.model.num_res_blocks)

    # Create a dummy state to load the checkpoint into
    dummy_input = jnp.ones([1, config.image_size, config.image_size, config.model.channels])
    dummy_r, dummy_t = jnp.ones([1]), jnp.ones([1])
    params = model.init(jax.random.PRNGKey(0), dummy_input, dummy_r, dummy_t)["params"]
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-4))

    # Load the trained checkpoint
    state = checkpoints.restore_checkpoint(FLAGS.checkpoint_dir, state)
    if not state:
        raise FileNotFoundError(f"No checkpoint found at {FLAGS.checkpoint_dir}")
    print(f"Restored checkpoint from step {state.step}")

    # Generate samples
    key = jax.random.PRNGKey(42)
    sample_shape = (FLAGS.num_samples, config.image_size, config.image_size, config.model.channels)
    samples = sample_fn(state, key, sample_shape)

    # Save the grid
    save_image_grid(samples, FLAGS.output_path)
    print(f"Saved {FLAGS.num_samples} generated samples to {FLAGS.output_path}")


if __name__ == "__main__":
    app.run(main)
