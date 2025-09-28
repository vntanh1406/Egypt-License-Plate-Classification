# Egyptian License Plate Classification

## Project Description
This project develops an automated classification system for Egyptian vehicle license plates based on color and format features, using a Convolutional Neural Network (CNN) with transfer learning from ResNet-50. The system classifies six main types of license plates:

- **Private** (Light Blue)
- **Taxi** (Orange) 
- **Commercial** (Red)
- **Public Transport** (Gray)
- **Diplomats** (Green)
- **Tourist Temporary** (Yellow)

The project was developed in Jupyter Notebook with PyTorch, handling imbalanced data using **WeightedRandomSampler** and optionally **Focal Loss**. It includes 5 models trained via cross-validation and an automated inference system that outputs results in CSV format.

## Installation Requirements

### Software and Versions
- **Python**: 3.11 or higher (3.11.13 recommended for notebook compatibility)
- **Operating System**: Windows, macOS, or Linux
- **Jupyter Notebook**: To open and run `.ipynb` files

### Required Libraries
Recommended libraries and versions:
- `torch==2.1.0` (PyTorch with GPU support)
- `torchvision==0.16.0` (for pretrained models)
- `numpy==1.26.0`
- `pandas==2.1.0`
- `scikit-learn==1.3.0` (for metrics and cross-validation)
- `matplotlib==3.7.0`
- `seaborn==0.12.0`

### Environment Setup
1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   venv\Scripts\activate     # On Windows
````

2. Install libraries from the requirements file:

   ```bash
   pip install -r requirements.txt
   ```

3. Check GPU availability (optional):

   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
   ```

### Hardware

* **GPU**: NVIDIA GPU with CUDA 11.8+ recommended for faster training
* **RAM**: Minimum 8GB, 16GB recommended for batch size 64
* **Storage**: Around 2–3GB for dataset, models, and output

## How to Run

### Step 1: Prepare Data

Ensure the `Data` directory contains data with the following structure:

* `train/`: Training images by class (Commercial/, Diplomats/, Private/, PublicTransport/, Taxi/, TouristTemporary/)
* `val/`: Validation images
* `test/`: Test images

### Step 2: Run the Notebook

1. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

2. Open `egypt-license-plate-classification-final.ipynb`

3. Run the cells sequentially:

   * **Import libraries and setup**: Cells 1–2
   * **Load and explore data**: Cells 3–6
   * **Data preprocessing and augmentation**: Cells 7–11
   * **Model definition**: Cells 12–13
   * **Training with cross-validation**: Cell 14
   * **Evaluation and metrics**: Cells 15–17
   * **Inference and prediction**: Cells 18–22

### Step 3: Results

* **Trained Models**: Available in `model/`

  * `best_model_fold1.pt` to `best_model_fold5.pt` (5-fold cross-validation)
* **Prediction Results**: Stored in `output/`

  * `predictions.csv`: CSV file in format `filename,predicted_plate_type,confidence`

## Directory Structure

```text
Egypt-License-Plate-Classification/
├── egypt-license-plate-classification-final.ipynb   # Main notebook
├── README.md                                       # This documentation
├── requirements.txt                                # Required libraries
├── Data/                                           # Dataset
│       ├── train/                                  # Training images (by class)
│       ├── val/                                    # Validation images
│       └── test/                                   # Test images
├── model/                                          # Trained models
│   ├── best_model_fold1.pt
│   ├── best_model_fold2.pt
│   ├── best_model_fold3.pt
│   ├── best_model_fold4.pt
│   └── best_model_fold5.pt
└── output/                                         # Prediction results
    ├── predictions.csv                             # CSV predictions
```

## Deliverables

**Completed:**

* **Trained Models**: `model/best_model_fold1.pt` to `best_model_fold5.pt` (5 PyTorch models)
* **Main Notebook**: `egypt-license-plate-classification-final.ipynb` (full pipeline)
* **Prediction Results**: `output/predictions.csv` (20 sample predictions with confidence scores)
* **Dependencies**: `requirements.txt` (library list)
* **Dataset Structure**: `Data/` with train/val/test split and challenging samples

**Output Structure:**

* CSV format: `filename,predicted_plate_type,confidence`
* Classes: `private, taxi, commercial, publictransport, diplomats, touristtemporary`

## Key Results

**Model Performance:**

* **Cross-validation**: Completed 5-fold, with separate models per fold
* **Output Classes**: Successfully classified into 6 categories

**Technical Specs:**

* **Framework**: PyTorch with ResNet-50 backbone
* **Data Handling**: WeightedRandomSampler for class imbalance
* **Models**: 5 trained models from cross-validation

## Troubleshooting

**Common Issues:**

* **ImportError**: Run `pip install -r requirements.txt` to install dependencies
* **CUDA issues**: Notebook can run on CPU (slower, set device='cpu')
* **Path errors**: Ensure `Data/train`, `Data/val`, `Data/test` directories exist
* **Memory errors**: Reduce batch_size in notebook if RAM is insufficient
* **Model loading**: Ensure `.pt` files in `model/` match the PyTorch version

**Tips:**

* Use Google Colab or Kaggle if no local GPU available
* Check `predictions.csv` to confirm output format
* Notebook has been tested with sample outputs included

## Additional Documentation

**Technical References:**

* **ResNet-50**: He, K., et al. (2016). *Deep Residual Learning for Image Recognition*
* **PyTorch Documentation**: [https://pytorch.org/docs/](https://pytorch.org/docs/)
* **Transfer Learning**: Pretrained ImageNet weights for computer vision

**Dataset Info:**

* Format: JPG images, organized by class folders
* Classes: 6 Egyptian license plate types with distinct colors

**Repository Stats:**

* Trained models: 5 files (~100MB+ each)
* Sample predictions: 20 images with high confidence
* Code: 1 comprehensive Jupyter notebook (378 lines)

## Important Notes

**About Dataset:**

* Repository includes dataset structure in `Data/`
* Contains train/val/test split
* Total 6 classes: Commercial, Diplomats, Private, Public Transport, Taxi, Tourist Temporary

**Usage:**

* Pretrained models available (5 models in `model/`)
* Supports direct inference or retraining from scratch
* Sample outputs included for reference

**Further Development:**

* Fine-tune with new data if available
* Adjust confidence threshold based on use case
* Extend classification to additional license plate types

## Authors

**Egypt License Plate Classification Project**
Completion Date: 27/09/2025
Framework: PyTorch + ResNet-50

---
*This repository contains a complete Egyptian license plate classification system, ready for use with pretrained models and real-world examples.*