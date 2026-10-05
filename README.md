# Waste Image Classification with PyTorch

A computer vision project that classifies waste images into six categories using **PyTorch** and **transfer learning with ResNet18**.

The project compares a custom CNN baseline with a pretrained ResNet18 model and evaluates performance using accuracy, precision, recall, F1-score, and a confusion matrix.

## Project Overview

The goal is to automatically classify waste images into:

* Cardboard
* Glass
* Metal
* Paper
* Plastic
* Trash

The project started with a custom convolutional neural network and achieved approximately **59% validation accuracy**. Transfer learning with a pretrained ResNet18 significantly improved performance, reaching **93.68% validation accuracy**.

## Dataset

The project uses the **TrashNet** image dataset.

Dataset characteristics:

* **2,527 images**
* **6 classes**
* 80% training / 20% validation split
* Stratified split to preserve class distribution
* Images resized to `224 × 224`

Class distribution:

| Class     |    Images |
| --------- | --------: |
| Cardboard |       403 |
| Glass     |       501 |
| Metal     |       410 |
| Paper     |       594 |
| Plastic   |       482 |
| Trash     |       137 |
| **Total** | **2,527** |

The dataset is not included in this repository.

## Approach

### 1. Custom CNN Baseline

A small CNN was initially developed from scratch using:

* Convolutional layers
* ReLU activation
* Max pooling
* Fully connected layers
* Cross-entropy loss
* Adam optimizer

The baseline achieved approximately:

**59% validation accuracy**

This provided a reference point for evaluating more advanced approaches.

### 2. Transfer Learning with ResNet18

A pretrained **ResNet18** model from `torchvision` was used as the main model.

The pretrained model was initialized with ImageNet weights.

Initially, the pretrained layers were frozen and the final classification layer was replaced to output six classes.

The final experiment fine-tuned the last ResNet block (`layer4`) while keeping the earlier feature extraction layers frozen.

The final classifier was changed from the original ImageNet classifier to:

```text
512 → 6
```

### Image preprocessing

Images were resized to `224 × 224` and normalized using ImageNet statistics:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

### Training configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Model         | ResNet18         |
| Input size    | 224 × 224        |
| Batch size    | 32               |
| Epochs        | 8                |
| Optimizer     | Adam             |
| Learning rate | 0.0001           |
| Loss function | CrossEntropyLoss |
| Hardware      | CPU              |

The best model checkpoint was saved based on validation accuracy.

## Results

### Final Validation Performance

**Validation Accuracy: 93.68%**

| Class            | Precision |   Recall | F1-score |
| ---------------- | --------: | -------: | -------: |
| Cardboard        |      0.97 |     0.96 |     0.97 |
| Glass            |      0.94 |     0.93 |     0.93 |
| Metal            |      0.88 |     0.98 |     0.92 |
| Paper            |      0.95 |     0.95 |     0.95 |
| Plastic          |      0.95 |     0.89 |     0.91 |
| Trash            |      0.92 |     0.89 |     0.91 |
| **Macro Avg**    |  **0.94** | **0.93** | **0.93** |
| **Weighted Avg** |  **0.94** | **0.94** | **0.94** |

### Baseline vs Final Model

| Model                       | Validation Accuracy |
| --------------------------- | ------------------: |
| Custom CNN                  |                ~59% |
| ResNet18 Feature Extraction |              80.24% |
| ResNet18 Fine-Tuning        |          **93.68%** |

Transfer learning provided a substantial improvement over the custom CNN baseline.

The most notable improvement was on difficult minority classes. For example, trash recall improved from approximately **7% with the custom CNN to 89% with the fine-tuned ResNet18**.

## Confusion Matrix

The final model produced the following confusion matrix:

```text
[[78, 0, 1, 2, 0, 0],
 [ 0,93, 3, 1, 3, 0],
 [ 0, 1,80, 0, 1, 0],
 [ 2, 0, 3,113, 1, 0],
 [ 0, 5, 3, 1,86, 2],
 [ 0, 0, 1, 2, 0,24]]
```

The model performs particularly well on cardboard, metal, and paper.

## Example Prediction

The prediction script returns the predicted class along with probabilities for all six classes.

Example:

```text
Prediction:
  trash

Class probabilities:
  trash       97.17%
  paper        1.45%
  cardboard    0.59%
  plastic      0.43%
  metal        0.31%
  glass        0.06%
```

The tested image was an unseen validation image and was correctly classified as `trash`.

## Project Structure

```text
waste-image-classifier/
│
├── data/
│   └── raw/
│       └── dataset-resized/
│
├── models/
│   └── waste_classifier_resnet18_finetuned.pth
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── outputs/
│   └── evaluation_results.txt
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

> The dataset and trained model weights are excluded from Git tracking because of file size. The model can be recreated by running the training pipeline.

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd waste-image-classifier
```

Create a virtual environment:

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Dataset Setup

Place the TrashNet dataset in:

```text
data/raw/dataset-resized/
```

The directory should contain:

```text
dataset-resized/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

## Training

From the project root:

```powershell
python src/train.py
```

The best validation checkpoint will be saved to:

```text
models/waste_classifier_resnet18_finetuned.pth
```

## Evaluation

Run:

```powershell
python src/evaluate.py
```

The script reports:

* Validation accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

Results are also saved to:

```text
outputs/evaluation_results.txt
```

## Prediction

To classify an individual image:

```powershell
python src/predict.py --image "path\to\image.jpg"
```

The script outputs the predicted class and probability for every waste category.

## Key Engineering Lessons

This project demonstrates several practical computer vision and deep learning concepts:

* Building and training a CNN from scratch
* PyTorch dataset and DataLoader pipelines
* Image preprocessing and normalization
* Train/validation splitting
* Transfer learning
* Fine-tuning pretrained models
* Model checkpointing
* Multi-class classification
* Precision, recall, and F1 evaluation
* Confusion matrix analysis
* Command-line image inference
* Reproducible project structure

## Future Improvements

Potential improvements include:

* Data augmentation with carefully selected transformations
* Hyperparameter tuning
* Learning-rate scheduling
* More extensive cross-validation
* GPU training
* Model export using TorchScript or ONNX
* REST API deployment
* Web interface for image classification
* Docker deployment

## License

This project is intended for educational and portfolio purposes.
