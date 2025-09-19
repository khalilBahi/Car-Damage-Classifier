# Car Damage Classifier

A deep learning-powered system for automatically classifying vehicle damage types from images. This project uses EfficientNetB0 as the base model to detect and classify six types of car damage: crack, dent, glass shatter, lamp broken, scratch, and tire flat.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Web Interface](#web-interface)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project addresses the challenge of automating vehicle damage assessment, which traditionally requires manual inspection by experts. The system can classify images into six damage categories:

- **Crack**: Surface cracks in body panels
- **Dent**: Indentations in metal surfaces
- **Glass Shatter**: Broken or cracked glass components
- **Lamp Broken**: Damaged headlights, taillights, or indicators
- **Scratch**: Surface scratches on paint or body
- **Tire Flat**: Deflated or damaged tires

## ✨ Features

- **High Accuracy**: Uses EfficientNetB0 with transfer learning for optimal performance
- **Real-time Inference**: Fast prediction on new images
- **Web Interface**: User-friendly Gradio interface for easy testing
- **Class Balancing**: Handles dataset imbalance with computed class weights
- **Comprehensive Evaluation**: Detailed metrics and visualization tools
- **Modular Design**: Well-organized scripts for different tasks

## 📊 Dataset

The project uses the CarDD (Car Damage Detection) dataset, which contains annotated images of vehicle damage. The dataset is organized as follows:

```
data/
├── raw/CarDD_release/CarDD_COCO/          # Original COCO format dataset
├── annotations_csv/                        # CSV annotations for each split
├── processed/                              # Processed images by class
├── train/                                  # Training set (80%)
├── val/                                    # Validation set (10%)
└── test/                                   # Test set (10%)
```

### Data Distribution
- **Total Classes**: 6 damage types
- **Split Ratio**: 80% train, 10% validation, 10% test
- **Image Format**: JPEG images resized to 224x224 pixels
- **Preprocessing**: EfficientNet-specific preprocessing applied

## 🏗️ Model Architecture

The model uses **EfficientNetB0** as the backbone with the following architecture:

```
EfficientNetB0 (ImageNet pretrained)
    ↓
Global Average Pooling 2D
    ↓
Batch Normalization
    ↓
Dense Layer (512 units, ReLU, L2 regularization)
    ↓
Dropout (0.3)
    ↓
Dense Layer (6 units, Softmax)
```

### Training Strategy
1. **Phase 1**: Train only the classification head (15 epochs)
2. **Phase 2**: Fine-tune the last 30 layers of EfficientNet (20 epochs)

### Key Features
- **Transfer Learning**: Leverages ImageNet pretrained weights
- **Class Weights**: Balanced training with computed class weights
- **Label Smoothing**: Reduces overconfidence (smoothing=0.1)
- **Data Augmentation**: Rotation, shifts, zoom, and brightness variations
- **Early Stopping**: Prevents overfitting with patience-based stopping

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/khalilBahi/Car-Damage-Classifier.git
cd Car-Damage-Classifier
```

2. **Create a virtual environment**:
```bash
python -m venv car_damage_env
source car_damage_env/bin/activate  # On Windows: car_damage_env\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install tensorflow>=2.10.0
pip install numpy pandas matplotlib seaborn
pip install scikit-learn pillow
pip install gradio
pip install pathlib
```

4. **Download the dataset**:
   - Download the CarDD dataset
   - Place it in `data/raw/CarDD_release/`

## 💻 Usage

### Quick Start

1. **Prepare the data**:
```bash
cd scripts
python convert_json_to_csv.py      # Convert COCO annotations to CSV
python process_images_from_csv.py   # Process and organize images
python data_split.py                # Split into train/val/test
```

2. **Train the model**:
```bash
python train_model.py
```

3. **Evaluate the model**:
```bash
python evaluate_model.py
```

4. **Launch web interface**:
```bash
cd ../web
python gradio_app.py
```

The web interface will be available at `http://localhost:7860`

### Using Pre-trained Models

If you have pre-trained models, place them in the `models/` directory:
- `best_model.keras`: Best model from training
- `final_model.keras`: Final model after all epochs
- `class_indices.json`: Class label mappings

## 📁 Project Structure

```
Car-Damage-Classifier/
├── data/                           # Data directory
│   ├── raw/                        # Original CarDD dataset
│   ├── annotations_csv/            # CSV format annotations
│   ├── processed/                  # Processed images by class
│   ├── train/                      # Training images
│   ├── val/                        # Validation images
│   └── test/                       # Test images
├── models/                         # Trained models and metadata
│   ├── best_model.keras           # Best performing model
│   ├── final_model.keras          # Final trained model
│   └── class_indices.json         # Class to index mapping
├── scripts/                        # Training and processing scripts
│   ├── convert_json_to_csv.py     # Convert COCO to CSV format
│   ├── process_images_from_csv.py  # Process images from CSV
│   ├── data_split.py              # Split data into train/val/test
│   ├── train_model.py             # Main training script
│   └── evaluate_model.py          # Model evaluation script
├── web/                           # Web interface
│   └── gradio_app.py              # Gradio web application
├── slides/                        # Presentation materials
│   └── car_damage_detection_presentation.md
└── README.md                      # This file
```

## 🎓 Training

### Training Process

The training script (`scripts/train_model.py`) implements a two-phase training approach:

1. **Phase 1 - Head Training** (15 epochs):
   - Freeze EfficientNetB0 backbone
   - Train only the classification head
   - Learning rate: 1e-3
   - Use class weights for balance

2. **Phase 2 - Fine-tuning** (20 epochs):
   - Unfreeze last 30 layers of backbone
   - Fine-tune with lower learning rate: 1e-5
   - Continue with class weights

### Key Parameters
- **Image Size**: 224x224 pixels
- **Batch Size**: 16 (training), 32 (evaluation)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy with label smoothing
- **Callbacks**: Early Stopping, Model Checkpoint, Reduce LR on Plateau

### Data Augmentation
- Rotation: ±15 degrees
- Width/Height Shift: ±10%
- Zoom: ±15%
- Horizontal Flip: Yes
- Brightness: 0.9-1.1 range

## 📈 Evaluation

The evaluation script provides comprehensive metrics:

- **Accuracy Score**: Overall classification accuracy
- **Confusion Matrix**: Class-wise performance visualization
- **Classification Report**: Precision, Recall, F1-score per class
- **Visual Analysis**: Confusion matrix heatmap

Run evaluation:
```bash
cd scripts
python evaluate_model.py
```

## 🌐 Web Interface

The Gradio web application provides an intuitive interface for testing the model:

### Features
- **Image Upload**: Drag and drop or click to upload
- **Real-time Prediction**: Instant classification results
- **Top Predictions**: Shows top 3 most likely classes with confidence scores
- **User-friendly**: Clean, responsive interface

### Launching the Interface
```bash
cd web
python gradio_app.py
```

The interface will be available at:
- **Local**: http://localhost:7860
- **Network**: http://0.0.0.0:7860 (accessible from network)
- **Public**: Gradio provides a temporary public link

## 📊 Results

### Model Performance
- **Training Accuracy**: ~XX% (varies by run)
- **Validation Accuracy**: ~XX% (varies by run)
- **Test Accuracy**: ~XX% (varies by run)

### Class-wise Performance
The model handles class imbalance through:
- Computed class weights (clipped between 0.5-3.0)
- Label smoothing (0.1)
- Balanced data augmentation

*Note: Specific performance metrics will be generated after training completion.*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CarDD Dataset**: Thanks to the creators of the Car Damage Detection dataset
- **EfficientNet**: Google's EfficientNet architecture for efficient image classification
- **TensorFlow/Keras**: Deep learning framework
- **Gradio**: For the easy-to-use web interface

## 📞 Contact

For questions or support, please open an issue in the GitHub repository.

---
