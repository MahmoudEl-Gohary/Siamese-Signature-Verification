# Siamese Neural Network for Signature Verification

A deep learning system for authenticating handwritten signatures using Siamese neural networks with contrastive learning. This project implements a sophisticated signature verification system capable of distinguishing between genuine and forged signatures. Developed for DSAI 308 - Deep Learning course.

## 🎯 Project Overview

This project addresses the critical problem of signature verification in security and authentication systems. Using a Siamese neural network architecture, the system learns similarity metrics between signature pairs to determine authenticity with accuracy.

## ✨ Key Features

- **Siamese Architecture**: Twin CNNs for feature extraction and comparison
- **Contrastive Learning**: loss function for similarity learning
- **Data Augmentation**: image augmentation pipeline
- **Balanced Dataset**: Automated positive and negative pair generation
- **Real-time Inference**: Optimized for signature verification applications
- **Comprehensive Evaluation**: Multiple performance metrics and visualization

## 🏗️ Architecture Overview

### Siamese Network Components

1. **Feature Extractor**: Convolutional Neural Network
   - Conv2D layers with ReLU activation
   - MaxPooling for dimensionality reduction
   - GlobalAveragePooling for feature aggregation
   - Dense embedding layer (128 dimensions)

2. **Distance Computation**: Manhattan distance metric
3. **Classification**: Sigmoid-activated dense layer for binary output

### Model Pipeline
```
Input Pair → Feature Extraction → Distance Calculation → Classification → Probability Score
```

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Image processing and computer vision
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **Scikit-learn** - Metrics and data splitting
- **ImageDataGenerator** - Data augmentation

## 📁 Project Structure

```
├── Project.ipynb          # Main implementation notebook
├── README.md                     # Project documentation
└── data.txt                      # Dataset link on kaggle
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/siamese-signature-verification-dsai308.git
cd siamese-signature-verification-dsai308
```

2. Prepare your dataset in the following structure:
```
Dataset/
├── dataset1/
│   ├── real/
│   └── forge/
├── dataset2/
│   ├── real/
│   └── forge/
└── ...
```

## 📊 Model Architecture Details

### Feature Extraction Network
```python
def siamese_model(input_shape, embedding_dim=128):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    pooledOutput = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(embedding_dim)(pooledOutput)
    model = Model(inputs, outputs)

    return model
```

### Contrastive Loss Function
```python
  def contrastive_loss(y_true, y_pred):
      square_pred = ops.square(y_pred)
      margin_square = ops.square(ops.maximum(margin - (y_pred), 0))
      return ops.mean((1 - y_true) * square_pred + (y_true) * margin_square)
```

### Distance Metric
```python
def manhattan_distance(vectors):
    x, y = vectors
    return tf.math.reduce_sum(tf.math.abs(x - y), axis=1, keepdims=True)
```

## 📈 Performance Results

### Model Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 73.8% |
| **Precision** | 70.2% |
| **Recall** | 82.5% |
| **F1-Score** | 75.9% |

### Training Configuration
- **Batch Size**: 64
- **Epochs**: 50 (with early stopping)
- **Learning Rate**: Adam optimizer with ReduceLROnPlateau
- **Image Size**: 224×224 pixels
- **Embedding Dimension**: 128

## 🔧 Data Processing Pipeline

### Image Preprocessing
1. **Loading**: Read signature images in grayscale
2. **Resizing**: Standardize to 224×224 pixels
3. **Normalization**: Scale pixel values to [0, 1]
4. **Augmentation**: Apply rotation, shift, shear, zoom, and flip transformations

### Pair Generation
- **Positive Pairs**: Combinations of genuine signatures from the same signer
- **Negative Pairs**: Genuine vs forged signature combinations
- **Balancing**: Equal numbers of positive and negative pairs
- **Shuffling**: Random order for training stability

### Data Augmentation Parameters
```python
datagen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
```
## 🚧 Challenges and Solutions

### Data Quality Issues
**Challenge**: Variability in handwriting styles and signature quality
**Solution**: Comprehensive data augmentation and normalization

### Model Training Challenges
**Challenge**: Overfitting and convergence issues
**Solution**: Early stopping, learning rate scheduling, and regularization

### Class Imbalance
**Challenge**: Unequal genuine/forged pair distribution
**Solution**: Automated balancing and stratified sampling

## 📝 Academic Context

**Course**: DSAI 308 - Deep Learning  
**Institution**: University of science and technology at zewailcity
**Semester**: Fall 2024  
**Team Members**: Ahmed Sameh (202201124), Mahmoud Elgohary (202201819)

## 📄 License

This project is created for educational purposes as part of coursework requirements.
---

*This project demonstrates the practical application of deep learning techniques in biometric authentication, showcasing skills in neural network design, contrastive learning, and computer vision for security applications.*
