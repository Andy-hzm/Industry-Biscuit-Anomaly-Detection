# Industrial Biscuit Defect Detection

Deep-learning quality control for an industrial biscuit line: detect defective
Tarallini biscuits from 256×256 photos, both by **classifying known defect types**
(supervised) and by **flagging anything that doesn't look normal** while training
only on good biscuits (unsupervised anomaly detection).

The project was built in two phases:

| Phase | Course | Focus |
|---|---|---|
| [Phase 1 — Spring 2025](phase1_spring2025/) | Bayesian Machine Learning | Reconstruction-based anomaly detection: Autoencoder, VAE, DCGAN, ViT-VAE |
| [Phase 2 — Fall 2025](phase2_fall2025/) | Computer Vision | Supervised defect classification (CNN, ResNet50V2, EfficientNet, MobileNetV3) and feature-based anomaly detection (ResNet18 + One-Class SVM, DDPM diffusion) |

## Dataset

[Industry Biscuit (Cookie) Dataset](https://www.kaggle.com/datasets/imonbilk/industry-biscuit-cookie-dataset/data)
— 1,225 biscuits, each photographed at 4 rotations (4,900 images, 256×256 RGB).

| Class | Biscuits |
|---|---|
| No defect | 474 |
| Defect: not complete (shape) | 465 |
| Defect: strange object | 158 |
| Defect: color | 128 |

The data is not included in this repo. Download it from Kaggle, then run the
notebooks in [`phase2_fall2025/00_data_prep/`](phase2_fall2025/00_data_prep/) to build the
`train / valid / test` folders the models expect.

## Results at a glance

**Unsupervised anomaly detection (OK vs. defective, trained on OK only)**

| Model | Phase | Test result |
|---|---|---|
| ResNet18 features + One-Class SVM | 2 | **AUC 0.933**, defect precision 0.90 / recall 0.80 / F1 0.85 |
| DCGAN (reconstruction error) | 1 | Accuracy 0.89, defect precision 0.92 / recall 0.86 / F1 0.89 |
| Convolutional VAE | 1 | Accuracy 0.79 |
| Convolutional Autoencoder | 1 | Accuracy 0.81, defect recall 0.56 |
| ViT-VAE (pretrained ViT-Base encoder) | 1–2 | AUC 0.61, F1 0.65 |
| DDPM diffusion (noise-prediction error) | 2 | Did not separate OK from defective |

**Supervised 4-class defect classification**

| Model | Test accuracy | Macro F1 |
|---|---|---|
| ResNet50V2 (fine-tuned) | ~100% | 1.00 |
| EfficientNet (fine-tuned) | ~99% | 0.99 |
| Basic CNN (Keras Tuner) | ~96% | 0.96 |
| Advanced CNN (BatchNorm + L2) | ~88% | 0.89 |

> ⚠️ The supervised split in phase 2 assigns images to train/valid/test individually,
> so rotated copies of the same biscuit can appear in both train and test. These
> accuracies are therefore likely optimistic; phase 1 split by biscuit (groups of 4)
> to avoid this.

**Takeaways.** Pretrained CNN features plus a simple one-class model were the most
reliable anomaly detector. High-capacity generative models (ViT-VAE, DDPM) reconstruct
defects *too* well — they "repair" stains and foreign objects — so reconstruction error
stops being a useful anomaly score.

## Repository layout

```
phase1_spring2025/          Spring 2025 notebooks + original write-up (README.md)
phase2_fall2025/
  00_data_prep/             Build train/valid/test folders from the Kaggle download
  01_eda/                   Class balance, sample grids, resolution analysis
  02_supervised/            Basic CNN, advanced CNN, transfer learning
  03_unsupervised/          ResNet18 + OCSVM, DDPM diffusion
docs/fall2025_report.docx   Phase 2 final report
```

Trained model weights (hundreds of MB each) are not included.

## Team

- **Phase 1:** Zimeng (Andy) Huang, [@mvemuri-22](https://github.com/mvemuri-22), [@alexlee0422](https://github.com/alexlee0422)
- **Phase 2:** Andy Huang, Qi Qiao, Tracy Xu, Wen Yao

Originally developed at [mvemuri-22/Industry-Biscuit-Anomaly-Detection](https://github.com/mvemuri-22/Industry-Biscuit-Anomaly-Detection) (phase 1); full commit history is preserved here.
