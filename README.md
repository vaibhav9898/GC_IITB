# GC_IITB - Bird Species Classification Challenge

A comprehensive machine learning project implementing multiple deep learning architectures for bird species classification. This repository contains implementations of state-of-the-art computer vision models including Vision Transformers, Swin Transformers, EfficientNet, ResNet, and custom CNN architectures for classifying 200 different bird species.

## 🎯 Project Overview

This project tackles the challenging task of fine-grained bird species classification using deep learning techniques. The goal is to accurately classify images into 200 distinct bird species using various neural network architectures and advanced training techniques.

### Key Features

- **Multiple Model Architectures**: Implementation of 5+ different deep learning models
- **Transfer Learning**: Leveraging pre-trained models for improved performance
- **Advanced Data Augmentation**: Including Mixup, CutMix, and Test Time Augmentation (TTA)
- **Sophisticated Loss Functions**: Center Loss for enhanced feature learning
- **Progressive Training**: Gradual unfreezing and learning rate scheduling
- **Early Stopping**: Preventing overfitting with patience-based stopping
- **Comprehensive Evaluation**: Multiple metrics including accuracy, AUC, and balanced accuracy

## 📊 Dataset

The project uses a bird classification dataset with the following structure:

- **Training Set**: ~12,000+ images across 200 bird species
- **Test Set**: ~2,800+ images for evaluation
- **Classes**: 200 distinct bird species (labels 1-200)
- **Data Format**: Images organized in class-specific folders

### Data Structure
```
data/
├── train/
│   ├── 001.Black_footed_Albatross/
│   ├── 002.Laysan_Albatross/
│   └── ... (200 species total)
└── test/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## 🏗️ Model Architectures

### 1. Vision Transformer (ViT)
- **File**: `vit.ipynb`
- **Architecture**: ViT Large Patch16 224
- **Features**: 
  - Pre-trained on ImageNet
  - Progressive unfreezing strategy
  - Test Time Augmentation (TTA)
  - Dropout regularization

### 2. Vision Transformer from Scratch
- **File**: `vit_scratch.ipynb` 
- **Architecture**: ViT Large trained from scratch
- **Features**:
  - No pre-trained weights
  - Custom training regime
  - Enhanced regularization

### 3. Swin Transformer
- **File**: `swin.py`
- **Architecture**: Swin-B Patch4 Window12 384
- **Features**:
  - Hierarchical vision transformer
  - Center Loss integration
  - Mixup and CutMix augmentation
  - 384×384 input resolution

### 4. EfficientNet B4
- **File**: `efficient_b4.ipynb`
- **Architecture**: EfficientNet-B4
- **Features**:
  - Compound scaling approach
  - Pre-trained weights
  - Optimized for efficiency

### 5. ResNet50
- **File**: `resnet50.ipynb`
- **Architecture**: ResNet-50
- **Features**:
  - Classic residual architecture
  - Proven performance baseline
  - Transfer learning approach

### 6. Custom CNN
- **File**: `cnn_scratch.ipynb`
- **Architecture**: Custom CNN from scratch
- **Features**:
  - Built from ground up
  - Multiple evaluation metrics
  - Balanced accuracy optimization

## 🚀 Getting Started

### Prerequisites

```bash
# Required Python packages
pip install torch torchvision
pip install timm
pip install pandas numpy matplotlib
pip install scikit-learn
pip install Pillow
pip install tqdm
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/vaibhav9898/GC_IITB.git
cd GC_IITB
```

2. Install dependencies:
```bash
pip install -r requirements.txt  # Create if needed
```

3. Prepare your data:
   - Place training images in `data/train/` organized by species folders
   - Place test images in `data/test/`
   - Ensure `train_labels.csv` contains image paths and labels

## 🔧 Usage

### Training Models

#### Vision Transformer
```bash
jupyter notebook vit.ipynb
# Run all cells to train ViT with progressive unfreezing
```

#### Swin Transformer
```bash
python swin.py
# Trains Swin-B model with advanced augmentation
```

#### EfficientNet B4
```bash
jupyter notebook efficient_b4.ipynb
# Trains EfficientNet-B4 with transfer learning
```

#### ResNet50
```bash
jupyter notebook resnet50.ipynb
# Baseline ResNet-50 implementation
```

### Model Configuration

Each model includes configurable parameters:

- **Batch Size**: Optimized for each architecture
- **Learning Rate**: Architecture-specific rates with scheduling
- **Epochs**: Early stopping with patience
- **Data Augmentation**: Customizable transforms
- **Input Resolution**: Model-optimized sizes

### Key Training Features

#### Progressive Unfreezing (ViT)
```python
# Unfreeze backbone layers after initial epochs
if epoch == unfreeze_epoch:
    for param in model.blocks.parameters():
        param.requires_grad = True
```

#### Test Time Augmentation
```python
# Multiple augmented predictions averaged
tta_transforms = [
    test_transform,
    horizontal_flip_transform,
    # ... additional transforms
]
```

#### Center Loss (Swin)
```python
# Enhanced feature learning
center_loss = CenterLoss(num_classes=200, feat_dim=1024)
total_loss = ce_loss + 0.003 * center_loss
```

## 📈 Results and Performance

### Model Comparison

| Model | Best Accuracy | Training Time | Parameters |
|-------|---------------|---------------|------------|
| ViT Large | TBA | ~4-6 hours | 304M |
| Swin-B | TBA | ~6-8 hours | 88M |
| EfficientNet-B4 | TBA | ~3-4 hours | 19M |
| ResNet50 | TBA | ~2-3 hours | 25M |
| Custom CNN | TBA | ~1-2 hours | Variable |

### Training Strategies

- **Data Augmentation**: RandomResizedCrop, ColorJitter, RandomRotation
- **Regularization**: Dropout, Label Smoothing, Weight Decay
- **Optimization**: AdamW optimizer with cosine annealing
- **Mixed Precision**: For faster training (where applicable)

## 📁 File Structure

```
GC_IITB/
├── README.md                    # This file
├── vit.ipynb                   # Vision Transformer (pretrained)
├── vit_scratch.ipynb          # Vision Transformer (from scratch)
├── vit_old.ipynb             # Legacy ViT implementation
├── swin.py                    # Swin Transformer implementation
├── efficient_b4.ipynb        # EfficientNet B4 model
├── resnet50.ipynb           # ResNet50 baseline
├── cnn_scratch.ipynb        # Custom CNN from scratch
├── train_labels.csv         # Training data labels
├── train.csv               # Training data paths
├── test.csv               # Test data paths
├── train_split.csv        # Training split
├── val_split.csv         # Validation split
├── submission.csv        # Competition submission
└── submission_Vaibhav_13.csv  # Specific submission file
```

## 🔬 Advanced Techniques

### Data Augmentation
- **Mixup**: Linear interpolation between images and labels
- **CutMix**: Cutting and pasting patches between images
- **Test Time Augmentation**: Multiple predictions on augmented test images

### Loss Functions
- **Cross-Entropy**: Standard classification loss
- **Center Loss**: Intra-class feature compactness
- **Label Smoothing**: Regularization technique

### Training Optimizations
- **Gradient Clipping**: Preventing exploding gradients
- **Learning Rate Scheduling**: Cosine annealing and step decay
- **Early Stopping**: Monitoring validation performance
- **Model Checkpointing**: Saving best performing models

## 📊 Evaluation Metrics

The project evaluates models using multiple metrics:

- **Standard Accuracy**: Overall classification accuracy
- **Balanced Accuracy**: Account for class imbalance
- **AUC Score**: Area under ROC curve
- **Mean Average Precision (mAP)**: Precision across classes
- **Sensitivity/Specificity**: Clinical evaluation metrics

## 💡 Tips for Best Results

1. **Data Preprocessing**: Ensure consistent image preprocessing
2. **Model Selection**: Choose architecture based on computational resources
3. **Hyperparameter Tuning**: Experiment with learning rates and batch sizes
4. **Ensemble Methods**: Combine multiple model predictions
5. **Cross-Validation**: Use robust validation strategies

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- New model implementations
- Performance improvements
- Bug fixes
- Documentation enhancements

## 📝 License

This project is available for educational and research purposes. Please check with the repository owner for specific licensing terms.

## 🙏 Acknowledgments

- **IIT Bombay**: For providing the challenge framework
- **Timm Library**: For pre-trained model implementations
- **PyTorch**: For the deep learning framework
- **Research Community**: For architectural innovations

## 📧 Contact

For questions or collaborations, please reach out through GitHub issues or contact the repository maintainer.

---

**Note**: This project represents a comprehensive exploration of modern computer vision techniques for fine-grained classification tasks. Results may vary based on computational resources and specific dataset characteristics.