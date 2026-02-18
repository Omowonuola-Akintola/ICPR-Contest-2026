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
├── classification.ipynb          # Main classification pipeline with MoCo + ResNet50
├── classification_Prithvi.ipynb  # Prithvi foundation model approach
├── ssl_moco_py.ipynb            # Self-supervised learning with MoCo v2
├── pyproject.toml               # Project dependencies
├── output/                      # Model logs and results
└── README.md                    
```

**Models & Approaches**  

**1. Self-Supervised Learning with MoCo v2**  
Notebook: `ssl_moco_py.ipynb`
- Backbone: ResNet50
- Technique: Momentum Contrast v2 (MoCo) with momentum encoder
- Input: 12 Sentinel-2 bands (B1-B9, B8A, B11, B12)
- Augmentations: Random resize, flips, Gaussian blur, brightness adjustment
- Memory Bank: 2048 negative samples

**2. Downstream Classification with MoCo Encoder**   
Notebook: `classification.ipynb`
- Fine-tunes ResNet50 pre-trained with MoCo on labeled data
- Input: 12 Sentinel-2 bands
- Loss: Focal loss for class imbalance
- Metrics: Macro F1, Accuracy
- Optimization: Hyperparameters tuned with Optuna (learning rate, batch size, weight decay, label smoothing)

**3. Foundation Model Approach (Prithvi)**   
Notebook: `classification_Prithvi.ipynb`  
- Uses Prithvi Vision Transformer, pre-trained on NASA HLS dataset
- Input: 6 Sentinel-2 bands (B2-B7)
- Framework: IBM Terratorch for geospatial AI
- Loss: Focal loss for class imbalance
- Metrics: Macro F1, Accuracy

**Results**  
Model Performance comparison:  
| Experiment | Weights | Val Macro F1-score (%) | Evaluation Score (%) |
|------------|---------|-----------------------|--------------------|
| Baseline   | ResNet50 S2 weights | 58.7 | 81.25 |
| Exp 1      | SSL pretrained subset | 67.8 | 87.5 |
| Exp 2      | SSL pretrained subset + aug* | 64.9 | 87.5 |
| Exp 3      | SSL pretrained full | 26.8 | 50.0 |
| Exp 4      | Prithvi 300M | 53.8 | 75.0 |
| Exp 5      | Prithvi 600M | 55.0 | 81.5 |

Observations:  
- SSL pre-training with MoCo significantly improves downstream classification performance.
- Prithvi offer strong generalization but may underperform on small, domain-specific labeled datasets compared to task-specific SSL approaches.

