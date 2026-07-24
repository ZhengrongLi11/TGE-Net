<h1 align="center">
  Geometry-Aligned Multimodal Generation with Evidential Discrimination for Text-Conditioned Brain T1CE MRI Synthesis
</h1>

## Overview

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/cf273664-3d5c-4c23-8cc5-f2bf4a5ba7a9"
    alt="TGE-Net framework"
    width="700"
  />
</p>

> We propose **Text-Conditioned Geometric Evidential Network (TGE-Net)**, a multimodal generative framework for lesion-aware T1CE MRI synthesis. TGE-Net integrates multimodal non-contrast MRI with lesion-related textual information to complement visual representations with lesion-specific semantic knowledge.
>
> To drive TGE-Net, we further propose two novel designs:
>
> **(i) Multimodal Geometric Alignment (MGA)** reduces representation drift across multimodal images and text by minimizing the Gramian-volume-based discrepancy among multimodal feature embeddings in a shared geometric space.
>
> **(ii) Fisher Evidential Discrimination (FED)** enables uncertainty-aware adversarial discrimination by converting patch-wise real/fake logits into Dirichlet evidence distributions, thereby providing reliable adversarial supervision.

## Datasets

Download the datasets through the official channels and rearrange the files according to the following structure. The dataset path can be modified in the `options/brats.yaml` file.

### BraTS2020

```text
BraTS2020_Train
├── flair
│   ├── BraTS20_Training_001_flair.nii
│   ├── BraTS20_Training_002_flair.nii
│   ├── BraTS20_Training_003_flair.nii
│   └── ...
├── t2
│   ├── BraTS20_Training_001_t2.nii
│   ├── BraTS20_Training_002_t2.nii
│   ├── BraTS20_Training_003_t2.nii
│   └── ...
├── t1
│   ├── BraTS20_Training_001_t1.nii
│   ├── BraTS20_Training_002_t1.nii
│   ├── BraTS20_Training_003_t1.nii
│   └── ...
└── t1ce
    ├── BraTS20_Training_001_t1ce.nii
    ├── BraTS20_Training_002_t1ce.nii
    ├── BraTS20_Training_003_t1ce.nii
    └── ...
```

### BraTS-PEDs

```text
BraTS2023_Train
├── flair
│   ├── BraTS23_Training_001_flair.nii.gz
│   ├── BraTS23_Training_002_flair.nii.gz
│   ├── BraTS23_Training_003_flair.nii.gz
│   └── ...
├── t2
│   ├── BraTS23_Training_001_t2.nii.gz
│   ├── BraTS23_Training_002_t2.nii.gz
│   ├── BraTS23_Training_003_t2.nii.gz
│   └── ...
├── t1
│   ├── BraTS23_Training_001_t1.nii.gz
│   ├── BraTS23_Training_002_t1.nii.gz
│   ├── BraTS23_Training_003_t1.nii.gz
│   └── ...
└── t1ce
    ├── BraTS23_Training_001_t1ce.nii.gz
    ├── BraTS23_Training_002_t1ce.nii.gz
    ├── BraTS23_Training_003_t1ce.nii.gz
    └── ...
```

## Usage

### Text Encoder

Download the pretrained text encoder from [Hugging Face](https://huggingface.co/pqt33/bert_model/tree/main), and place the downloaded files in the `bert_model/` directory according to the following structure.

```text
bert_model/
├── Bio_ClinicalBERT/
│   ├── config.json
│   ├── flax_model.msgpack
│   └── ...
├── bert_config.py
├── med.py
└── TextEncoder.py
```

### Train

Edit the `options/brats.yaml` file for training configuration and run the following command to train.

```bash
python train.py options/brats.yaml
```

### Test

Edit the `options/brats.yaml` file for testing configuration and run the following command to test.

```bash
python test.py options/brats.yaml
```

## Results

The qualitative visualization results of TGE-Net are shown below.

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/e5c1f819-11d6-47d8-a177-bda488fe2389"
    alt="Fig 3"
    width="700"
  />
</p>

<p align="center">
  <em>Qualitative comparison of T1CE MRI synthesis results.</em>
</p>
