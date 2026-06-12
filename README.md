# Vision Transformer (ViT)

A Vision Transformer implementation for image classification on brain MRI scans.

## Repository Structure

```
├── best_model.pth          # Trained model checkpoint
├── dataset.py              # Dataset loading and preprocessing
├── train.py                # Training script
├── test.py                 # Testing and evaluation script
├── pyproject.toml          # Project configuration
├── dataset/
│   ├── Training/           # Training dataset
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/            # Testing dataset
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
├── models/                 # Model components
│   ├── vit.py              # Vision Transformer model
│   ├── block.py            # Transformer block
│   ├── create_patches.py   # Patch creation module
│   ├── embeddings.py       # Patch embedding module
│   ├── mlp.py              # MLP component
│   ├── multi_head_attention.py  # Multi-head attention module
│   ├── self_attention.py   # Self-attention module
│   └── __init__.py
└── docs/
    └── notes.md            # Architecture documentation
```

## Dataset

The dataset contains brain MRI scan images organized into four classification categories:
- Glioma
- Meningioma
- No tumor
- Pituitary

Images are organized into separate `Training` and `Testing` directories.

## Model Architecture

The Vision Transformer model implements the following components:

- **Patch Creation**: Reshapes input images into a sequence of flattened 2D patches
- **Patch Embedding**: Embeds patches into a fixed embedding dimension
- **Class Token**: Learnable embedding prepended to the sequence
- **Positional Embedding**: Learnable 1D position embeddings added to patch embeddings
- **Transformer Blocks**: Multiple transformer blocks consisting of:
  - Multi-head self-attention
  - MLP blocks
  - Layer normalization applied before each block
  - Residual connections after each block
- **Classification Head**: Linear layer for 4-class classification

## Configuration

Model parameters (from `train.py`):
- Image dimensions: 224×224
- Patch size: 16
- Embedding dimension: 32
- Number of attention heads: 4
- Number of transformer blocks: 4
- Number of classes: 4

Training parameters:
- Number of epochs: 100
- Learning rate: 1e-4
- Optimizer: AdamW with weight decay (0.03)
- Loss function: CrossEntropyLoss
- Scheduler: Cosine annealing learning rate scheduler
- Batch size: 8

## Dependencies

- torch >= 2.12.0
- torchvision >= 0.27.0
- tqdm
- matplotlib (for dataset visualization)

## Usage

### Training

Run the training script:
```bash
python train.py
```

The script will:
- Load the training and testing datasets
- Initialize the Vision Transformer model
- Train for 100 epochs
- Save the best model to `best_model.pth`

### Testing

Evaluate the model on the test set:
```bash
python test.py
```

The script loads the checkpoint from `best_model.pth` and evaluates performance on the test dataset.

### Dataset Exploration

Visualize dataset samples:
```bash
python dataset.py
```

## Device Support

The model automatically uses Intel XPU if available, otherwise defaults to CPU for training and inference.
