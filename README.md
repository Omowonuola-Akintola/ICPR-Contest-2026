# ICPR-Contest-2026  

**Background**  
Competition: *Beyond Visible Spectrum: AI for Agriculture 2026*
Task: *Boosting automatic crop disease classification using Sentinel-2 satellite imagery and self-supervised learning (SSL).*  

This project focuses on automatic crop disease classification from multispectral Sentinel-2 data. The challenge is to extract meaningful features from limited labeled data, using unlabeled satellite imagery using self-supervised learning, and evaluating downstream classification performance.

**Objectives**   

The main goal of this project is to improve crop disease classification accuracy by combining self-supervised pre-training and downstream supervised classification:  
- Self-Supervised Learning (SSL): Pre-train feature representations from unlabeled Sentinel-2 images using a SSL model.
- Downstream Classification: Fine-tune SSL-pretrained encoders on the labeled crop disease dataset.

**Dataset**  

Dataset was provided by the ICPR contest here: https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026p2/data.  
The key features include: 
- **Unlabeled Data:** Sentinel-2 (S2A) time-series imagery, organized by location and acquisition time
- **Labeled Data:** 900 images for training of the following classes `Aphid, Blast, RPH, Rust`
- **evaluation set:** 40 unlabelled samples for evaluating model performance

**Project Structure**  

```
ICPR-Contest-2026/
├── classification.ipynb          # SSL + classification pipeline with MoCo
├── pyproject.toml               # Project dependencies
├── output/                      # Model logs and results
├── report.pdf                   # detailed report on the project
└── README.md                    
```

**Tasks:**  
**1. Self-Supervised Learning with MoCo v2**  
- Backbone: ResNet50
- Technique: Momentum Contrast v2 (MoCo) with momentum encoder
- Input: 12 Sentinel-2 bands (B1-B9, B8A, B11, B12)
- Augmentations: Random resize, flips, Gaussian blur, brightness adjustment
- Memory Bank: 2048 negative samples

**2. Downstream Classification with MoCo Encoder**   
- Fine-tunes ResNet50 pre-trained with MoCo on labeled data
- Input: 12 Sentinel-2 bands
- Loss: Focal loss for class imbalance
- Metrics: Macro F1, Accuracy
- Optimization: Hyperparameters tuned with Optuna (learning rate, batch size, weight decay, label smoothing)


**Results**  
Model Performance comparison:  
| Experiment | Weights | Val Macro F1-score (%) | Evaluation Score (%) |
|------------|---------|-----------------------|--------------------|
| Baseline   | ResNet50 S2 weights | 58.7 | 81.25 |
| Exp 1      | SSL pretrained subset | 68.7 | 87.5 |
| Exp 2      | SSL pretrained full | 64.9 | 81.5 |

Observations:  
- SSL pre-training with MoCo significantly improves downstream classification performance.

