import tensorflow as tf
import tensorflow_datasets as tfds


def get_cifar10_dataset(batch_size):
    """Loads and prepares the CIFAR-10 dataset."""

    def preprocess(features):
        image = tf.cast(features["image"], tf.float32)
        # Normalize to [-1, 1]
        image = (image / 127.5) - 1.0
        return image

    # Ensure TF does not grab GPU memory
    tf.config.experimental.set_visible_devices([], "GPU")

    ds = tfds.load("cifar10", split="train", shuffle_files=True)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(10000)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return tfds.as_numpy(ds)
