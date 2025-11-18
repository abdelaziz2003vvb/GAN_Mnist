# GAN for MNIST Digit Generation

A simple Generative Adversarial Network (GAN) implementation in PyTorch for generating handwritten digits from the MNIST dataset.

## Overview

This project implements a basic GAN architecture with:
- **Generator**: Transforms random noise into fake digit images
- **Discriminator**: Distinguishes between real and fake images
- Training on the MNIST handwritten digit dataset

## Architecture

### Generator
- Input: 64-dimensional latent vector (noise)
- Hidden layer: 256 units with LeakyReLU
- Output: 784 dimensions (28×28 flattened image) with Tanh activation

### Discriminator
- Input: 784-dimensional flattened image
- Hidden layer: 128 units with LeakyReLU
- Output: Single probability value with Sigmoid activation

## Requirements

```
torch
torchvision
tensorboard
```

Install with:
```bash
pip install torch torchvision tensorboard
```

## Usage

Run the training script:
```bash
python gan_mnist.py
```

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir=runs
```

## Hyperparameters

- **Learning Rate**: 3e-4
- **Latent Dimension**: 64
- **Batch Size**: 32
- **Epochs**: 50
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy (BCE)

## Training Process

1. **Discriminator Training**:
   - Classifies real images as real (label = 1)
   - Classifies generated (fake) images as fake (label = 0)
   - Loss = average of real and fake losses

2. **Generator Training**:
   - Generates fake images from random noise
   - Tries to fool discriminator into classifying fakes as real
   - Loss based on discriminator's output for fake images

## Output

- Training logs printed every epoch
- TensorBoard visualization of:

  - Generated (fake) images over time
  - Real images from dataset
- Images saved in `runs/GAN_MNIST/` directory

## Key Features

- **Normalization**: Images normalized to [-1, 1] range to match Tanh output
- **Fixed noise**: Same noise vector used throughout training to track generator improvement
- **Separate optimizers**: Independent optimization for generator and discriminator
- **Detached gradients**: Fake images detached during discriminator training to prevent generator updates

## Notes

- Model automatically uses GPU if available (CUDA)
- Dataset downloads automatically to `dataset/` folder on first run
- Training progress can be monitored in real-time via TensorBoard

## Results

After training, the generator should produce realistic-looking handwritten digits from random noise. Check TensorBoard to see the quality improvement over epochs.
