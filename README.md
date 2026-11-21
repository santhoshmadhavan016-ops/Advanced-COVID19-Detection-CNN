# COVID-19 Detection from Chest Images Using Deep Learning

A deep learning-based medical image classification project that identifies four categories of chest images using three different convolutional neural network models: Custom CNN, ResNet50, and DenseNet121.

## 1. Project Overview

This project focuses on the classification of chest images into four categories:

- COVID
- Normal
- Viral Pneumonia
- Lung Opacity

Three deep learning architectures are implemented and evaluated using the same test dataset. Their performance is compared using accuracy, precision, recall, F1-score, and ROC-AUC.

The project also incorporates Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize image regions that influence model predictions.

**Note:** This is an academic research project and is not intended for clinical diagnosis or medical decision-making.

## 2. Dataset Information

The dataset contains **21,165 images** distributed across four classes.

| Class | Number of Images |
|---|---:|
| COVID | 3,616 |
| Normal | 10,192 |
| Viral Pneumonia | 1,345 |
| Lung Opacity | 6,012 |
| **Total** | **21,165** |

### Dataset Split

| Dataset | Images | Percentage |
|---|---:|---:|
| Training | 14,816 | 70% |
| Validation | 3,174 | 15% |
| Testing | 3,175 | 15% |
| **Total** | **21,165** | **100%** |

The dataset is organized into class-specific folders containing images and masks.

The current data loader uses the image files for classification. Mask-based training is not implemented in the verified pipeline.

## 3. Deep Learning Models

Three architectures are used in this project.

### 3.1 Custom CNN

A custom Convolutional Neural Network is developed for four-class image classification.

Main components:
- Convolutional layers
- Feature extraction
- Pooling operations
- Fully connected classification layers
- Softmax output layer

### 3.2 ResNet50

ResNet50 is a pretrained deep learning architecture based on residual connections.

Implementation highlights:
- ImageNet pretrained weights
- Grayscale-to-RGB input conversion
- ResNet50 preprocessing
- Global Average Pooling
- Dense classification layer
- Dropout regularization
- Four-class softmax output

### 3.3 DenseNet121

DenseNet121 is implemented using transfer learning with ImageNet pretrained weights.

Implementation highlights:
- ImageNet pretrained DenseNet121
- Grayscale-to-RGB input conversion
- DenseNet preprocessing
- Frozen pretrained feature extractor
- Global Average Pooling
- Dense classification layer
- Dropout regularization
- Four-class softmax output

## 4. Project Structure

```text
Advanced-COVID19-Detection-CNN/
│
├── data/
│   ├── dataset/
│   │   ├── COVID/
│   │   ├── Normal/
│   │   ├── Viral Pneumonia/
│   │   └── Lung_Opacity/
│   └── dataset_splits.npz
│
├── models/
│   ├── custom_cnn_best.keras
│   ├── custom_cnn_final.keras
│   ├── resnet50_best.keras
│   ├── resnet50_final.keras
│   ├── densenet121_best_v2.weights.h5
│   └── densenet121_final_v2.weights.h5
│
├── src/
│   ├── 01_config.py
│   ├── 02_data_loader.py
│   ├── 03_data_visualization.py
│   ├── 04_cnn_architecture.py
│   ├── 05_train.py
│   ├── 06_evaluate.py
│   ├── 07_gradcam.py
│   ├── 08_resnet50_architecture.py
│   ├── 09_resnet50_train.py
│   ├── 10_resnet50_evaluate.py
│   ├── 11_densenet121_architecture.py
│   ├── 11_resnet50_gradcam.py
│   ├── 12_densenet121_train.py
│   ├── 13_evaluate_densenet121.py
│   ├── 14_densenet121_gradcam.py
│   ├── 15_compare_models_roc.py
│   └── 16_model_comparison.py
│
├── logs/
├── results/
│   ├── custom_cnn/
│   ├── resnet50/
│   ├── densenet121/
│   ├── comparison/
│   └── old_results_backup/
│
├── experiments/
├── notebooks/
├── requirements.txt
├── README.md
└── .gitignore
```

## 5. Data Preprocessing

The images are processed using TensorFlow.

Preprocessing steps:

1. Read image files.
2. Decode PNG images.
3. Resize images to 224 × 224 pixels.
4. Convert images to grayscale with one channel.
5. Normalize pixel values to the range [0, 1].
6. Create TensorFlow datasets.
7. Batch the images and use prefetching for efficient loading.

**Configuration:**

| Parameter | Value |
|---|---|
| Image Size | 224 × 224 |
| Channels | 1 |
| Batch Size | 8 |
| Number of Classes | 4 |
| Random Seed | 42 |
| Configured Learning Rate | 0.0001 |

The pretrained ResNet50 and DenseNet121 architectures internally convert grayscale images to three channels and apply their respective preprocessing functions.

## 6. Model Training

Each architecture is trained and evaluated separately.

Training features include:

- Adam optimizer
- Sparse categorical cross-entropy loss
- Class weighting to address class imbalance
- Model checkpointing
- Early stopping
- Learning rate reduction on plateau
- CSV training logs

The best validation-loss checkpoints are used for model evaluation.

### Training Commands

Run the following commands from the project root directory after activating the virtual environment.

**Custom CNN**
```bash
python src/05_train.py
```

**ResNet50**
```bash
python src/09_resnet50_train.py
```

**DenseNet121**
```bash
python src/12_densenet121_train.py
```

## 7. Model Evaluation

All three models are evaluated using the same test dataset containing 3,175 images.

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- ROC-AUC Score
- Sensitivity
- Specificity

### Evaluation Commands

**Custom CNN**
```bash
python src/06_evaluate.py
```

**ResNet50**
```bash
python src/10_resnet50_evaluate.py
```

**DenseNet121**
```bash
python src/13_evaluate_densenet121.py
```

## 8. Model Performance Comparison

The following table presents the experimental results obtained from the three models.

| Performance Metric | Custom CNN | ResNet50 | DenseNet121 |
|---|---:|---:|---:|
| Accuracy | 91.81% | 91.81% | 87.09% |
| Macro Precision | 0.9254 | 0.9258 | 0.8658 |
| Macro Recall | 0.9315 | 0.9213 | 0.8907 |
| Macro F1-score | 0.9283 | 0.9235 | 0.8774 |
| Macro ROC-AUC | 0.9875 | 0.9871 | 0.9746 |

### Performance Analysis

- **Custom CNN:** Achieved the highest macro recall, macro F1-score, and macro ROC-AUC.
- **ResNet50:** Achieved the highest macro precision.
- **DenseNet121:** Achieved 87.09% accuracy and 0.9746 macro ROC-AUC.
- Custom CNN and ResNet50 achieved the same test accuracy of 91.81%.

Based on these experimental results, the Custom CNN demonstrated the strongest overall performance across the reported macro recall, F1-score, and ROC-AUC metrics.

## 9. ROC-AUC Analysis

The project uses a multiclass One-vs-Rest ROC-AUC approach.

### Class-wise ROC-AUC Comparison

| Class | Custom CNN | ResNet50 | DenseNet121 |
|---|---:|---:|---:|
| COVID | 0.9991 | 0.9950 | 0.9775 |
| Normal | 0.9761 | 0.9775 | 0.9592 |
| Viral Pneumonia | 0.9987 | 0.9989 | 0.9986 |
| Lung Opacity | 0.9760 | 0.9768 | 0.9631 |

The macro ROC-AUC is calculated as the average of the four class-wise ROC-AUC scores.

## 10. Grad-CAM Explainability

Grad-CAM is used to visualize important image regions contributing to a model's classification output.

The project generates heatmaps for the four image categories using the trained models.

Grad-CAM helps interpret model predictions by highlighting regions that contribute to the selected class score.

The visualizations are saved in the corresponding model result folders.

## 11. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### Step 2: Navigate to the Project

```bash
cd Advanced-COVID19-Detection-CNN
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Environment

Windows:

```bash
.venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Prepare the Dataset

Place the dataset in the `data/dataset/` directory using the four class folders.

Ensure that the existing dataset split file is available at:

```text
data/dataset_splits.npz
```

### Step 7: Run the Project

Use the training and evaluation commands provided in the corresponding sections.

## 12. Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Deep Learning
- Convolutional Neural Networks
- Transfer Learning