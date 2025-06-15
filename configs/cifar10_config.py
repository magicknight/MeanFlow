import ml_collections


def get_config():
    """Gets the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # General
    config.dataset = "cifar10"
    config.image_size = 32

    # Training
    config.batch_size = 128
    config.num_epochs = 200
    config.learning_rate = 2e-4
    config.warmup_steps = 5000
    config.log_every_steps = 100
    config.save_every_steps = 1000

    # Model (U-Net)
    config.model = ml_collections.ConfigDict()
    config.model.channels = 3
    config.model.dim = 128
    config.model.dim_mults = (1, 2, 2, 2)
    config.model.num_res_blocks = 2

    # MeanFlow specific
    config.meanflow = ml_collections.ConfigDict()
    config.meanflow.loss_p = 0.75  # p=0.75 from paper ablation
    config.meanflow.rt_dist = "lognorm"  # 'uniform' or 'lognorm'
    config.meanflow.rt_sampler_params = {"mu": -2.0, "sigma": 2.0}
    config.meanflow.r_is_not_t_ratio = 0.75  # Ratio of samples where r != t

    return config
