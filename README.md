# Geometry-Aligned Multimodal Generation with Evidential Discrimination for Text-Conditioned Brain T1CE MRI Synthesis

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
