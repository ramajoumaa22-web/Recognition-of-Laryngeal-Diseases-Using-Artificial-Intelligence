# Laryngoscopy Disease Classification Using Deep Learning

## Overview

This project presents a deep learning pipeline for automatic classification of laryngeal diseases from laryngoscopy images.

Two independent experiments were conducted:

- **RGB Dataset** (original images)
- **Enhanced RGB Dataset** (image-enhanced version)

Each experiment trains and evaluates multiple state-of-the-art deep learning backbones independently without Knowledge Distillation.

---

## Dataset

The model classifies the following eight laryngeal diseases:

- Cyst
- Suspicion of Malignancy
- Laryngeal Cancer
- Leukoplakia
- Nodule
- Papilloma
- Polyp
- Reinke's Edema

The dataset is organized into:

```
Dataset/
│
├── train/
├── validation/
└── test/
```

Each folder contains one subfolder for every disease class.

---

## Features

- Multi-class medical image classification
- Comparison between original RGB and enhanced RGB images
- Independent training for multiple backbone architectures
- LoRA (Low-Rank Adaptation) implementation
- MixUp data augmentation
- Extensive image augmentation
- Class-balanced training
- Grad-CAM visualization support
- Captum explainability support
- Confusion matrix evaluation
- Performance comparison across models

---

## Deep Learning Models

The pipeline is designed to evaluate multiple modern CNN and Vision Transformer backbones independently.

Each model is trained separately under identical settings for fair comparison.

---

## Image Preprocessing

Training includes several preprocessing and augmentation techniques:

- Resize
- Random Crop
- Random Horizontal Flip
- Random Rotation
- Color Jitter
- Image Normalization
- MixUp Augmentation

The enhanced dataset is used to evaluate the effect of image enhancement on classification performance.

---

## Technologies

- Python
- PyTorch
- TorchVision
- timm
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow
- tqdm
- Captum
- Grad-CAM

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
```

Install the required packages:

```bash
pip install torch torchvision timm matplotlib seaborn scikit-learn pillow tqdm grad-cam captum
```

---

## Project Structure

```
.
├── laryngoscopy 2_RGB.ipynb
├── laryngoscopy 2_Enhancement.ipynb
├── RGB/
│   ├── train/
│   ├── validation/
│   └── test/
├── RGB_Enhanced/
│   ├── train/
│   ├── validation/
│   └── test/
└── README.md
```

---

## Results

The project evaluates classification performance on both the original RGB dataset and the enhanced RGB dataset, allowing comparison of the impact of image enhancement on disease recognition.

Evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Grad-CAM visualizations

---

## Explainability

To improve model interpretability, the project supports:

- Grad-CAM
- Captum

These techniques help visualize the image regions influencing model predictions.

---

## Future Work

Possible future improvements include:

- Additional medical image enhancement techniques
- Ensemble learning
- Knowledge Distillation
- Self-supervised pretraining
- Clinical deployment optimization

---

## License

This project is intended for research and educational purposes.
