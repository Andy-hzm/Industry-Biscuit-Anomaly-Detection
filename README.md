# Industrial Biscuit Defect Detection

Can a camera on a production line catch a broken biscuit, a red stain, or a screw baked
into the dough, ideally without anyone first labelling thousands of defects?

This project attacks the problem two ways:

- **Supervised classification** tells you *which* defect it is.
- **One-class anomaly detection** learns only what a good biscuit looks like and flags
  anything else.

The models compared range from a plain autoencoder to GANs, Vision-Transformer VAEs and
diffusion models.

<p align="center"><img src="docs/figures/pipeline.svg" width="100%" alt="Pipeline: dataset → biscuit-level split → supervised and one-class tracks"></p>

The work was done in two phases:

- **Spring 2025:** reconstruction-based anomaly detection (Bayesian ML course).
- **Fall 2025:** supervised classification and feature-based anomaly detection (Computer Vision course).

## Data

The [Industry Biscuit dataset](https://www.kaggle.com/datasets/imonbilk/industry-biscuit-cookie-dataset/data)
has 1,225 Tarallini biscuits, each photographed at 4 rotations (4,900 images, 256 × 256).
The classes are 474 OK, 465 broken shape, 158 foreign object and 128 color defect.

<p align="center"><img src="docs/figures/sample_images.jpg" width="60%" alt="Three examples each of OK, shape, object and color defects"></p>

Because each biscuit appears four times, splits must be made **per biscuit**. Otherwise a
rotated copy of a test biscuit sits in the training set.
[`prepare_data.py`](phase2_fall2025/00_data_prep/prepare_data.py) enforces this.

## How the anomaly score works

Every one-class model is trained on OK biscuits only and turns an image into a single
"how unusual is this?" number. The models differ in *where* that number comes from:

<p align="center"><img src="docs/figures/anomaly_scoring.svg" width="100%" alt="Reconstruction-based vs. feature-based anomaly scoring"></p>

## Results

<p align="center"><img src="docs/figures/results_f1.svg" width="85%" alt="F1 on the defective class: DCGAN 0.89, ResNet18 + OCSVM 0.85, VAE 0.82, autoencoder 0.70, ViT-VAE 0.65, DDPM no separation"></p>

### 1. Pretrained features separate defects; a generative latent space doesn't

The left panel shows ResNet18 features of validation biscuits: OK and defective form
distinct regions, so a One-Class SVM fit on OK alone reaches **AUC 0.93**. The right panel
shows the ViT-VAE latent space, where defects are scattered among the normal biscuits.

<p align="center"><img src="docs/figures/latent_spaces.png" width="100%" alt="t-SNE of ResNet18 features (separated) vs. ViT-VAE latent space (mixed)"></p>

Those same features organize the defects too. K-means on the defective biscuits' features,
with no labels involved, puts screws, bolts, washers, burns and red stains each in a group
of their own. (Clustering just the OCSVM's flags gives the same picture.)

<p align="center"><img src="docs/figures/ocsvm_clusters.png" width="70%" alt="Five K-means clusters of defective biscuits in ResNet18 feature space, one row per cluster"></p>

### 2. For reconstruction models, a *worse* reconstruction is a better detector

The convolutional VAE is too blurry to redraw red stains, so stained biscuits come back with
a large error, which is exactly the signal you want:

<p align="center"><img src="docs/figures/vae_reconstructions.png" width="80%" alt="VAE: originals on top, blurry reconstructions below; stains are not reproduced"></p>

The ViT-VAE, with a pretrained ViT-Base encoder, redraws screws, washers and burn marks
almost perfectly. The defect disappears from the error map, and test AUC drops to 0.61.

<p align="center"><img src="docs/figures/vitvae_reconstructions.png" width="100%" alt="ViT-VAE: defective biscuits (top) reconstructed faithfully, defects included (bottom)"></p>

The DDPM diffusion model does the opposite: it "repairs" defects, fading red stains and
washers into the biscuit. The defects show up locally in the error map but barely move the
image-wide error (columns: original · reconstruction · error).

<p align="center"><img src="docs/figures/ddpm_reconstructions.png" width="90%" alt="DDPM original, reconstruction and error map for an OK biscuit, a stained biscuit and two with washers"></p>

The DCGAN struck the best balance: sharp enough to keep fine detail, but unable to draw
what it never saw. It scored **F1 0.89** on the defective class.

<p align="center"><img src="docs/figures/dcgan_ranked.png" width="90%" alt="DCGAN: highest-scoring images contain coins, bolts and washers; lowest-scoring are clean biscuits"></p>

### 3. Supervised classification

| Model | Test accuracy | Macro F1 |
|---|---|---|
| ResNet50V2 (fine-tuned) | ~100% | 1.00 |
| EfficientNet (fine-tuned) | ~99% | 0.99 |
| Basic CNN (Keras Tuner) | ~96% | 0.96 |
| Advanced CNN (BatchNorm + L2) | ~88% | 0.89 |

The misclassifications are informative: most are small washers or coins lying *beside* the
biscuit, which a classifier trained on whole-image appearance reads as OK.

<p align="center"><img src="docs/figures/resnet_misses.png" width="100%" alt="Foreign-object biscuits misclassified as OK: the object sits at the image edge"></p>

## Takeaways

- **For a production line, start with pretrained features + a one-class model.** It needs
  no defect labels, gets AUC 0.93, adapts to a new product by refitting on fresh OK images,
  and its flags group naturally into defect types for root-cause analysis.
- **Reconstruction-based detection needs a model of limited capacity.** The stronger the
  generator (ViT-VAE, DDPM), the better it hides the anomaly.
- **Use supervised classification once defect types are known and labelled.** It's the
  right tool then, but evaluate it on a biscuit-level split and watch for small objects
  near the image border.

## Run it

```bash
pip install -r requirements.txt

# Unzip the Kaggle download to data/IndustryBiscuit (Images/ + Annotations.csv), then:
python phase2_fall2025/00_data_prep/prepare_data.py --raw data/IndustryBiscuit --out data/DS_multiclass --mode multiclass
python phase2_fall2025/00_data_prep/prepare_data.py --raw data/IndustryBiscuit --out data/DS --mode ok_nok
```

Notebooks read from `data/` by default. Set `BISCUIT_DATA_DIR` to point elsewhere, for
example a mounted Google Drive in Colab. A GPU is recommended for everything except the EDA.

```
phase1_spring2025/            Autoencoder, VAE (+ defect clustering), DCGAN, ViT-VAE
                              README.md there has the full Spring 2025 write-up
phase2_fall2025/
  00_data_prep/               prepare_data.py: biscuit-level splits
  01_eda/                     class balance, pixel statistics, resolution study
  02_supervised/              basic CNN, advanced CNN, transfer learning
  03_unsupervised/            ResNet18 + One-Class SVM, DDPM diffusion
docs/figures/                 figures and diagrams used above
```

## Team

- **Spring 2025:** Zimeng (Andy) Huang, [@mvemuri-22](https://github.com/mvemuri-22), [@alexlee0422](https://github.com/alexlee0422)
- **Fall 2025:** Andy Huang, Qi Qiao, Tracy Xu, Wen Yao

Phase 1 was first published at
[mvemuri-22/Industry-Biscuit-Anomaly-Detection](https://github.com/mvemuri-22/Industry-Biscuit-Anomaly-Detection);
its full commit history is preserved here.
