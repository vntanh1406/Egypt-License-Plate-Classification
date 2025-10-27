# Egyptian License Plate Classification System

## 1. Project Description
This project develops an **automated classification system for Egyptian vehicle license plates** based on their **color-coded background**, a standard feature in Egypt's vehicle registration system. The model classifies license plates into **six official categories**:

- **Private** (Blue)  
- **Taxi** (Orange)  
- **Commercial** (Red)  
- **Public Transport** (Grey)  
- **Diplomatic** (Green)  
- **Tourist/Temporary** (Yellow)

The system uses **cropped license plate images** extracted from real-world Egyptian traffic videos. The classification is formulated as a **6-class supervised learning task** using deep learning, with **color** being the primary discriminative feature.

Three models were evaluated using **5-fold stratified cross-validation**:
- **Simple CNN** (custom lightweight architecture)  
- **EfficientNet-B0** (pretrained, transfer learning)  
- **MobileNetV2** (pretrained, lightweight baseline)

**Simple CNN achieved the best performance** with near-perfect accuracy (0.99 average, 1.00 best fold) and fastest training time.

> **Author**: Vo Ngoc Tram Anh  
> **Date**: October 27, 2025

---

## 2. Dataset Info

### 2.1. Data Sources
The dataset was manually extracted from **24 real-world YouTube videos** of Egyptian traffic, covering diverse conditions (day/night, urban/tourist areas, weather, lighting). Videos were selected to ensure natural class distribution and environmental variability.

**Video List**:
1. [https://www.youtube.com/watch?v=dGcEfxSG4eM](https://www.youtube.com/watch?v=dGcEfxSG4eM)
2. [https://www.youtube.com/watch?v=6v8kbz8ZxlM](https://www.youtube.com/watch?v=6v8kbz8ZxlM)
3. [https://www.youtube.com/watch?v=vIXW23oYxlA](https://www.youtube.com/watch?v=vIXW23oYxlA)
4. [https://www.youtube.com/watch?v=z3KQSdc3vQg](https://www.youtube.com/watch?v=z3KQSdc3vQg)
5. [https://www.youtube.com/watch?v=uYionr1X-Jw](https://www.youtube.com/watch?v=uYionr1X-Jw)
6. [https://www.youtube.com/watch?v=OqfWD2XGSZ0](https://www.youtube.com/watch?v=OqfWD2XGSZ0)
7. [https://www.youtube.com/watch?v=n3Bv5JmGsYQ](https://www.youtube.com/watch?v=n3Bv5JmGsYQ)
8. [https://www.youtube.com/watch?v=hvS5RC2P664](https://www.youtube.com/watch?v=hvS5RC2P664)
9. [https://www.youtube.com/watch?v=1RZ2dmkV2Rg](https://www.youtube.com/watch?v=1RZ2dmkV2Rg)
10. [https://www.youtube.com/watch?v=r31DFG5idLc](https://www.youtube.com/watch?v=r31DFG5idLc)
11. [https://www.youtube.com/watch?v=00xbFDbSZy0](https://www.youtube.com/watch?v=00xbFDbSZy0)
12. [https://www.youtube.com/watch?v=tY2oeBh2LZc](https://www.youtube.com/watch?v=tY2oeBh2LZc)
13. [https://www.youtube.com/watch?v=SbmpvXdqHHI](https://www.youtube.com/watch?v=SbmpvXdqHHI)
14. [https://www.youtube.com/watch?v=MRCXeannNac](https://www.youtube.com/watch?v=MRCXeannNac)
15. [https://www.youtube.com/watch?v=cv7w_dJq8pg](https://www.youtube.com/watch?v=cv7w_dJq8pg)
16. [https://www.youtube.com/watch?v=EcidnZfrpUg](https://www.youtube.com/watch?v=EcidnZfrpUg)
17. [https://www.youtube.com/watch?v=rqGLsZvOCCU](https://www.youtube.com/watch?v=rqGLsZvOCCU)
18. [https://www.youtube.com/watch?v=VaTQXfKE6Pw](https://www.youtube.com/watch?v=VaTQXfKE6Pw)
19. [https://www.youtube.com/watch?v=25DlLwARtlA](https://www.youtube.com/watch?v=25DlLwARtlA)
20. [https://www.youtube.com/watch?v=SuuFB8thgFo](https://www.youtube.com/watch?v=SuuFB8thgFo)
21. [https://www.youtube.com/watch?v=dlbH9d64xGo](https://www.youtube.com/watch?v=dlbH9d64xGo)
22. [https://www.youtube.com/watch?v=xExj4__wBKc](https://www.youtube.com/watch?v=xExj4__wBKc)
23. [https://www.youtube.com/watch?v=xYYQ5z9TyNs](https://www.youtube.com/watch?v=xYYQ5z9TyNs)
24. [https://www.youtube.com/watch?v=UwyKoqMxpyg](https://www.youtube.com/watch?v=UwyKoqMxpyg)

### 2.2. Frame Extraction & Labeling
- **Manual extraction**: Each video was watched in full; frames were captured only when a clear license plate was visible.
- **Quality filtering**: Blurry, occluded, or overexposed frames were discarded.
- **Context-aware labeling**: Ambiguous cases (e.g., orange under red lighting) were resolved using video context.

### 2.3. Raw Dataset Distribution (Before Augmentation)
| Folder | Category | Color | Count |
|-------|----------|-------|-------|
| `1_Private_Blue` | Private | Blue | 1219 |
| `2_Taxi_Orange` | Taxi | Orange | 348 |
| `3_Commercial_Red` | Commercial | Red | 239 |
| `4_PublicTransport_Grey` | Public | Grey | 131 |
| `5_Diplomats_Green` | Diplomatic | Green | 193 |
| `6_TouristTemporary_Yellow` | Tourist/Temporary | Yellow | 141 |

### 2.4. Train-Test Split
- **Test set**: 35 images **per class** (210 total) — **balanced**
- **Training set**: Remaining images -> augmented to **600 per class (except Blue: 1149)**

### 2.5. Data Augmentation (Albumentations)
Applied **only to minority classes** to reach ~600 samples/class:

| Transformation | Range | Probability | Purpose |
|----------------|-------|-------------|--------|
| `Rotate` | ±7° | 0.5 | Camera angle |
| `HorizontalFlip` | — | 0.3 | Spatial diversity |
| `RandomScale` | ±10% | 0.3 | Distance variation |
| `RandomBrightnessContrast` | ±0.02 | 0.3 | Mild lighting |

> **Color-preserving**: No hue/saturation shifts to avoid cross-class confusion.

**Final Training Set (After Augmentation)**:  
- Blue: 1149  
- Others: ~600 each → **Total ~4243**

**Test Set**: 35 × 6 = **210 images** (balanced)

---

## 3. Directory Structure
```text
Egypt-License-Plate-Classification/
├── SOL_ELP.ipynb                   # Main training & evaluation notebook
├── DA_ELP.ipynb                    # Data analysis & augmentation notebook
├── EgyptLicensePlates.pdf          # Detailed Report
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── augmented_data/                 # Final dataset
│   ├── train/
│   │   ├── 1_Private_Blue/
│   │   ├── 2_Taxi_Orange/
│   │   ├── 3_Commercial_Red/
│   │   ├── 4_PublicTransport_Grey/
│   │   ├── 5_Diplomats_Green/
│   │   └── 6_TouristTemporary_Yellow/
│   └── test/                       # 35 images per class
├── model/                          # Trained model weights (.pth)
│   ├── best_SimpleCNN_fold1.pth
│   ├── best_SimpleCNN_fold2.pth
│   ├── best_SimpleCNN_fold3.pth
│   ├── best_SimpleCNN_fold4.pth
│   ├── best_SimpleCNN_fold5.pth
│   ├── best_EfficientNetB0_fold1.pth
│   ├── ... (5 per model)
│   └── best_MobileNetV2_fold5.pth
└── test_predictions.csv            # Optional: predictions on test set
```

> **Note**: Model weights are saved in **`.pth` format** using `torch.save(model.state_dict(), ...)` — contains **only the learned weights**, not the full model object. Use `model.load_state_dict(torch.load(path))` to load.

---

## 4. Installation

### 4.1. Requirements
```bash
pip install -r requirements.txt
```

**`requirements.txt`**:
```txt
torch>=1.9.0
torchvision
numpy
scikit-learn
matplotlib
seaborn
pandas
opencv-python
albumentations
jupyter
```

### 4.2. Hardware
- **GPU**: Recommended (CUDA-enabled NVIDIA)
- **RAM**: ≥8GB
- **Storage**: ~2GB (dataset + models)

---

## 5. Usage

### 5.1. Run Training & Evaluation
1. Open `SOL_ELP.ipynb`
2. Update paths if needed:
   ```python
   DATA_DIR = "augmented_data/train"
   TEST_DIR = "augmented_data/test"
   ```
3. Run all cells:
   - 5-fold training with **early stopping** (patience=5)
   - Saves **best model per fold** → `.pth`
   - Evaluates on **held-out test set**
   - Generates: classification report, confusion matrix, misclassified samples

### 5.2. Generate `test_predictions.csv`
Add this after evaluation:
```python
import pandas as pd
df = pd.DataFrame({
    'filename': test_filenames,
    'true_label': [class_names[t] for t in y_true],
    'predicted_label': [class_names[p] for p in y_pred],
    'confidence': confidences
})
df.to_csv('test_predictions.csv', index=False)
```

---

## 6. Model Performance (5-Fold CV + Test Set)

| Model | Avg Accuracy | Best Accuracy | Avg F1 | Training Time/Fold |
|-------|--------------|---------------|--------|---------------------|
| **Simple CNN** | **0.99** | **1.00** | **0.99** | **1.37 min** |
| EfficientNet-B0 | 0.98 | 0.99 | 0.98 | 2.95 min |
| MobileNetV2 | 0.95 | 0.97 | 0.95 | 2.73 min |

> **Simple CNN is the best model**: highest accuracy, fastest training, most efficient.

---

## 7. Error Analysis (Simple CNN)

### Common Confusion Pairs:
- `Grey → Blue` (lighting desaturation)
- `Orange → Red` (hue adjacency under shadow)
- `Yellow → Grey/Orange` (overexposure)

### Key Insights:
- Errors are **systematic**, not random
- High-confidence mistakes due to **illumination artifacts**
- Model fails on **extreme lighting**, not representation

---

## 8. Key Results Summary

| Metric | Simple CNN | EfficientNet-B0 | MobileNetV2 |
|-------|------------|------------------|-------------|
| **Best Model** | Yes | No | No |
| **Accuracy (Avg)** | 0.99 | 0.98 | 0.95 |
| **F1-Score (Avg)** | 0.99 | 0.98 | 0.95 |
| **Training Time** | 1.37 min/fold | 2.95 min/fold | 2.73 min/fold |
| **Input Size** | 48×48 | 48×48 | 48×48 |

---

## 9. Limitations & Future Work

### Limitations:
- Sensitive to **extreme lighting** (glare, shadow, overexposure)
- **Color adjacency** (Orange vs Red, Grey vs Blue)
- Small input size (48×48) limits texture cues

### Future Work:
- Add **color jitter + histogram equalization**
- Use **HSV/Lab color space** preprocessing
- Add **attention mechanisms**
- Expand dataset with **night, rain, occlusion** cases

---

## 10. Deliverables

- **15 trained models** (`*.pth` weights)
- **Full pipeline**: `SOL_ELP.ipynb`
- **Data analysis**: `DA_ELP.ipynb`
- **Augmented dataset**: `augmented_data/`
- **Prediction export**: `test_predictions.csv` (optional)
- **Documentation**: This `README.md`

---

## 11. Notes on `.pth` Files

- **`.pth` = PyTorch state dictionary** (only weights, not model architecture)
- To load:
  ```python
  model = SimpleCNN()  # or EfficientNetB0/MobileNetV2
  model.load_state_dict(torch.load('model/best_SimpleCNN_fold1.pth'))
  model.eval()
---
