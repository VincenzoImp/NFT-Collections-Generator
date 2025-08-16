# NFT Collections Generator

A comprehensive deep learning toolkit for generating unique NFT collections using advanced Generative Adversarial Networks (GANs). This project combines multiple GAN architectures to create high-quality, diverse digital artwork suitable for NFT collections.

## 🎯 Overview

This project implements a multi-stage pipeline for NFT generation that leverages three different GAN architectures:

1. **DCGAN (Deep Convolutional GAN)** - Base image generation at 64x64 resolution
2. **SRGAN (Super-Resolution GAN)** - Upscaling to 256x256 with enhanced details
3. **DCGAN-2D** - Dual-discriminator architecture for style blending between collections

The pipeline produces high-quality, unique digital artworks by combining generative modeling with image enhancement techniques.

## 🏗️ Architecture

### Core Components

#### 1. DCGAN Generator
- **Input**: 100-dimensional noise vector
- **Architecture**: 5-layer transposed convolutional network
- **Output**: 64x64x3 RGB images
- **Features**: 
  - Batch normalization for stable training
  - ReLU activations with Tanh output
  - Progressive upsampling from 4x4 to 64x64

#### 2. DCGAN Discriminator
- **Input**: 64x64x3 RGB images
- **Architecture**: 5-layer convolutional network
- **Features**:
  - LeakyReLU activations
  - Batch normalization (except first layer)
  - Binary classification (real/fake)

#### 3. SRGAN Super-Resolution
- **Purpose**: Enhance 64x64 images to 256x256
- **Generator**: ResNet-based with sub-pixel convolution
- **Loss Function**: Combination of:
  - Adversarial loss
  - Perceptual loss (VGG-based)
  - Mean squared error
  - Total variation loss

#### 4. DCGAN-2D (Dual Collection Blending)
- **Innovation**: Two discriminators for style mixing
- **Purpose**: Generate art that blends characteristics from two different collections
- **Training**: Alternating optimization with weighted loss functions

## 📁 Project Structure

```
NFT-Collections-Generator/
├── models/
│   ├── DCGAN/
│   │   └── DCGAN.ipynb          # Base GAN implementation
│   ├── SRGAN/
│   │   └── SRGAN.ipynb          # Super-resolution enhancement
│   └── DCGAN-2D/
│       └── DCGAN-2D.ipynb       # Dual-collection blending
├── generateNFT.ipynb            # Complete pipeline execution
├── README.md
└── images/                      # Training data directory
    ├── real/
    │   ├── 64/
    │   │   ├── Collection1/
    │   │   └── Collection2/
    │   └── 128/
    │       └── Collection2/
    │           └── EAPES/
```

## 🚀 Getting Started

### Prerequisites

```bash
# Core dependencies
pip install torch torchvision
pip install numpy matplotlib opencv-python
pip install tqdm ipywidgets
pip install Pillow
```

### Training Pipeline

#### 1. Prepare Your Dataset
```bash
# Organize images in the following structure:
images/real/64/YourCollection/
└── [your training images here]
```

#### 2. Train Base DCGAN
```python
# In models/DCGAN/DCGAN.ipynb
modelDCGAN = DCGAN(
    dataroot='../../images/real/64/YourCollection',
    logfolder='output_folder',
    num_epochs=50,
    batch_size=128,
    image_size=64
)
img_list, G_losses, D_losses = modelDCGAN.train()
```

#### 3. Train SRGAN for Super-Resolution
```python
# In models/SRGAN/SRGAN.ipynb
# Prepare high-resolution dataset at 128x128
train_set = TrainDatasetFromFolder(
    '../../images/real/128/YourCollection',
    crop_size=88,
    upscale_factor=4
)
```

#### 4. Generate NFTs
```python
# In generateNFT.ipynb - Complete pipeline
# 1. Generate base image with DCGAN
# 2. Apply noise reduction
# 3. Upscale with SRGAN
# 4. Final enhancement
```

### Generation Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `batch_size` | Training batch size | 128 | 32-256 |
| `num_epochs` | Training epochs | 50 | 10-200 |
| `lr` | Learning rate | 0.0002 | 0.0001-0.01 |
| `nz` | Noise dimension | 100 | 50-200 |
| `ngf/ndf` | Feature map size | 64 | 32-128 |

## 🎨 Advanced Features

### Multi-Collection Blending (DCGAN-2D)

Create unique art by blending styles from two different collections:

```python
modelDCGAN_2D = DCGAN_2D(
    dataroot1='../../images/real/64/Collection1',
    dataroot2='../../images/real/64/Collection2',
    weight1=0.5,  # Balance between collections
    weight2=0.5
)
```

### Quality Enhancement Pipeline

The complete generation process includes:

1. **Base Generation**: DCGAN creates 64x64 initial artwork
2. **Noise Reduction**: OpenCV denoising for cleaner images
3. **Super-Resolution**: SRGAN upscales to 256x256 with detail enhancement
4. **Final Polish**: Additional denoising and formatting

### Loss Function Innovation

**DCGAN-2D Discriminator Fusion**:
```python
# Multiple fusion strategies implemented:
output = torch.min(output1, output2)    # Conservative approach
# output = torch.max(output1, output2)  # Aggressive approach  
# output = (output1 + output2) / 2      # Balanced approach
```

## 📊 Training Monitoring

### Loss Tracking
- Generator and Discriminator losses automatically logged
- Visual progress saved at regular intervals
- Training curves plotted for analysis

### Quality Metrics
- Real vs. Fake image comparisons
- Progressive generation examples
- Loss convergence analysis

## 🔧 Customization

### Model Architecture
Easily modify network architectures by adjusting:
- Layer depths and feature map sizes
- Activation functions and normalization
- Skip connections and residual blocks

### Training Strategy
- Adaptive learning rates
- Loss weighting for different objectives
- Custom data augmentation pipelines

## 💡 Use Cases

- **NFT Collections**: Generate thousands of unique digital artworks
- **Art Style Transfer**: Blend characteristics from different art styles
- **Data Augmentation**: Expand training datasets for other ML projects
- **Creative Exploration**: Experiment with AI-generated art concepts

## 🚨 Important Notes

### Hardware Requirements
- **GPU**: CUDA-compatible GPU recommended (8GB+ VRAM)
- **RAM**: 16GB+ system memory for large batch sizes
- **Storage**: Sufficient space for datasets and generated outputs

### Training Tips
- Start with smaller datasets to validate pipeline
- Monitor discriminator/generator balance during training
- Experiment with different loss weightings for style control
- Use checkpointing for long training runs

## 🛠️ Troubleshooting

### Common Issues

**Mode Collapse**: 
- Reduce discriminator learning rate
- Increase generator training frequency
- Add noise to discriminator inputs

**Training Instability**:
- Lower learning rates for both networks
- Implement gradient clipping
- Use spectral normalization

**Memory Issues**:
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

## 📈 Performance Optimization

- **Mixed Precision**: Use `torch.cuda.amp` for faster training
- **Data Loading**: Optimize `num_workers` for your system
- **Batch Size**: Balance between memory usage and training stability
- **Model Pruning**: Remove unnecessary parameters for inference

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- New GAN architectures (StyleGAN, Progressive GAN)
- Advanced loss functions
- Better evaluation metrics
- Web interface for easy generation
