# MeanFlow: A JAX Implementation

This repository provides a high-fidelity JAX implementation of the paper **"MeanFlow: A Principled and Effective Framework for One-Step Generative Modeling"**.

The paper introduces a novel framework for one-step generative modeling by modeling the *average velocity* of a probability flow, rather than the instantaneous velocity used in traditional Flow Matching or Diffusion Models. This is achieved through the **MeanFlow Identity**, a principled equation derived from first principles that relates the average and instantaneous velocities.

This implementation aims to be a clear, concise, and faithful reproduction of the core ideas, making it easy for researchers and practitioners to understand, use, and extend the MeanFlow framework.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [MeanFlow: A JAX Implementation](#meanflow-a-jax-implementation)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [How It Works: The Core Idea](#how-it-works-the-core-idea)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training a Model](#training-a-model)
    - [Generating Samples](#generating-samples)
  - [Code Structure](#code-structure)
  - [What's Next? (Roadmap)](#whats-next-roadmap)
    - [**1. Core Logic \& Structure (What you have now)**](#1-core-logic--structure-what-you-have-now)
    - [**2. Making it a Usable Project (Next Steps)**](#2-making-it-a-usable-project-next-steps)
    - [**3. Advanced Features (Long-term)**](#3-advanced-features-long-term)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)

## Key Features

- **Principled Approach:** The training objective is derived directly from the mathematical definition of average velocity, avoiding the need for complex heuristics or curriculum learning.
- **Efficient 1-NFE Sampling:** Generates high-quality samples with a single network evaluation, drastically reducing inference time compared to multi-step methods.
- **JAX-Powered:** The implementation leverages JAX for its performance and functional programming paradigm, featuring:
  - `jax.jvp` for efficient computation of the MeanFlow Identity.
  - `jax.jit` for just-in-time compilation of training and sampling loops.
  - `flax.linen` for a clear and robust neural network definition.
- **Educational:** The code is heavily commented to serve as a learning resource for understanding the paper's core concepts.

## How It Works: The Core Idea

Traditional flow-based models learn the instantaneous velocity `v(z, t)` and approximate an integral to generate a sample. This is slow and requires many steps.

MeanFlow learns the **average velocity** `u(z, r, t)` over a time interval `[r, t]`. To avoid calculating a slow integral during training, it uses the **MeanFlow Identity**:

$$
u(z_t, r, t) = v(z_t, t) - (t - r)\frac{d}{dt}u(z_t, r, t)
$$

The model `u_θ` is trained to satisfy this identity. The term `(d/dt)u` is computed efficiently using `jax.jvp`.

Once trained, generating a sample is as simple as asking the model for the average velocity over the entire path (`r=0`, `t=1`) and taking a single step:

$$
x_{generated} = \text{noise} - u_\theta(\text{noise}, r=0, t=1)
$$

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/magicknight/MeanFlow.git
    cd MeanFlow
    ```

2.  Create a virtual environment and install the required packages. For GPU/TPU support, please follow the official [JAX installation guide](https://github.com/google/jax#installation).

    ```bash
    pip install -r requirements.txt
    ```

## Usage

The core logic is contained in `main.py` (or a similar entry-point file you will create).

### Training a Model

To train a MeanFlow model, you will need a dataset and a configuration file.

```bash
# (Example command)
python train.py --config=configs/cifar10_config.py --workdir=./checkpoints/cifar10_meanflow
```

The training script will handle:
1.  Loading the dataset.
2.  Initializing the `MeanFlowModel` and optimizer.
3.  Running the training loop, which uses the `meanflow_loss_fn` to update the model.
4.  Saving checkpoints to the specified work directory.

### Generating Samples

Use the sampling script with a trained checkpoint to generate images.

```bash
# (Example command)
python sample.py --checkpoint=./checkpoints/cifar10_meanflow/checkpoint_100 --num_samples=64 --output_dir=./samples
```

This will load the trained model, generate samples in a single step (1-NFE), and save them to the output directory.

## Code Structure

```
MeanFlow/
├── configs/                # Configuration files for different datasets/models.
│   └── cifar10_config.py
├── src/
│   ├── model.py            # Definition of the MeanFlowModel (e.g., U-Net, DiT).
│   ├── training.py         # Contains the core meanflow_loss_fn and train_step.
│   ├── sampling.py         # Contains the 1-NFE sampling logic.
│   └── data.py             # Dataloading utilities.
├── train.py                # Main script to start a training run.
├── sample.py               # Main script to generate samples from a checkpoint.
├── requirements.txt        # Python package dependencies.
└── README.md               # This file.
```

## What's Next? (Roadmap)

This is the most important part to make your repository a living project.

### **1. Core Logic & Structure (What you have now)**

-   [x] **`main.py` (or similar):** A single file containing the core logic (model, loss, training, sampling). This is a great starting point.
-   [x] **`README.md`:** This file.
-   [x] **`requirements.txt`:** A file listing dependencies (`jax`, `flax`, `optax`, `numpy`, etc.).

### **2. Making it a Usable Project (Next Steps)**

-   **Modularization:** Break down the single `main.py` into a structured project. This is critical for usability and extensibility.
    -   `src/model.py`: Define the neural network architecture (e.g., a U-Net or DiT).
    -   `src/training.py`: Place the `meanflow_loss_fn` and `train_step` here.
    -   `src/sampling.py`: Place the `sample_fn` here.
    -   `src/data.py`: Create dataloaders for standard datasets (e.g., CIFAR-10, CelebA).
    -   `train.py`: A script that imports from `src/` and runs the training.
    -   `sample.py`: A script for generating samples from a checkpoint.

-   **Configuration System:** The paper ablates many hyperparameters (`r, t` samplers, loss metrics, etc.). You need a way to manage these.
    -   Create a `configs/` directory.
    -   Use a simple Python file for each config (e.g., `configs/cifar10_base.py`) that defines all hyperparameters. The popular `ml_collections` library is perfect for this.

-   **Pre-trained Models:** Train a model on a standard dataset like CIFAR-10 and upload the checkpoint to the repository (or a hosting service). This allows users to test sampling immediately without training.

-   **Visualizations & Logging:**
    -   Integrate with a logging framework like [TensorBoard](https://www.tensorflow.org/tensorboard) or [Weights & Biases](https://wandb.ai/) to track loss curves and view generated image samples during training.
    -   Create a simple Jupyter Notebook (`examples/demo.ipynb`) that loads a pre-trained model and generates a grid of images. This is the best way to showcase your work.

### **3. Advanced Features (Long-term)**

-   **Implement Ablations:** Add flags or config options to easily reproduce the paper's ablation studies (e.g., different time samplers, loss powers `p`, CFG).
-   **Support for DiT Architecture:** Implement the Vision Transformer (DiT) architecture from the paper to replicate the ImageNet results.
-   **Classifier-Free Guidance (CFG):** Implement the CFG version of the MeanFlow loss as described in Section 3.2.
-   **Contribute to Hugging Face:** Integrate your model with the Hugging Face `diffusers` library to make it widely accessible.

## Citation

If you find this work useful in your research, please consider citing the original paper:

```bibtex
@article{meanflow2024,
  title={MeanFlow: A Principled and Effective Framework for One-Step Generative Modeling},
  author={...},
  journal={arXiv preprint arXiv:...},
  year={2024}
}
```
*(Note: Please update with the correct author list and arXiv ID once available.)*

## Acknowledgements

This code is based on the methods described in the MeanFlow paper. We thank the original authors for their contribution to the field.