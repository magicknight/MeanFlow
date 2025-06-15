import jax
import jax.numpy as jnp
from PIL import Image
import numpy as np


@jax.jit
def sample_fn(state, key, shape):
    """Generates images in a single step (1-NFE)."""
    e = jax.random.normal(key, shape)

    r_sample = jnp.zeros((shape[0],))
    t_sample = jnp.ones((shape[0],))

    u_pred = state.apply_fn({"params": state.params}, e, r_sample, t_sample)
    x_generated = e - u_pred

    return x_generated


def save_image_grid(images, path, grid_size=(8, 8)):
    """Saves a grid of images."""
    # De-normalize from [-1, 1] to [0, 255]
    images = np.array(((images + 1) / 2) * 255, dtype=np.uint8)

    num_images, h, w, c = images.shape
    grid_img = Image.new("RGB", (grid_size[1] * w, grid_size[0] * h))

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            idx = i * grid_size[1] + j
            if idx < num_images:
                grid_img.paste(Image.fromarray(images[idx]), (j * w, i * h))

    grid_img.save(path)
