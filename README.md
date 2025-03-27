# Face Identification using ViT, FaceNet, and Ensemble

## Introduction

This repository contains the code and models for a face identification system developed for the [Fawry Kaggle Competition](https://www.kaggle.com/competitions/surveillance-for-retail-stores/overview). The competition tasked participants with building a system to match images of staff faces to their corresponding identities based on image similarity, rather than traditional classification. The system needed to be scalable and adaptable to unseen identities, making it suitable for real-world applications. Our solution leverages two powerful models—Vision Transformer (ViT) and FaceNet—combined through an ensemble method, achieving **3rd place** in the competition.

## Models

We fine-tuned two state-of-the-art models for face recognition and combined them into an ensemble:

- **ViT (Vision Transformer)**:
  - **Base Model**: `vit_base_patch16_224` from the `timm` library, pre-trained on ImageNet.
  - **Fine-Tuning**: Adapted with an embedding layer consisting of a linear transformation from 768 to 2048 dimensions, followed by batch normalization, PReLU activation, dropout (0.5), and a final linear layer to 1024-dimensional embeddings. Aggressive augmentations were applied during fine-tuning to enhance robustness (e.g., random resized cropping, larger coarse dropout to simulate occlusions like sunglasses).
  - **Training**: Utilized a combination of CrossEntropyLoss with label smoothing, TripletMarginLoss, ContrastiveLoss, and CurricularFace loss to optimize embeddings for similarity tasks.

- **FaceNet**:
  - **Base Model**: `InceptionResnetV1` from `facenet-pytorch`, pre-trained on VGGFace2.
  - **Fine-Tuning**: Extended with an embedding layer transforming 512-dimensional features to 1024 dimensions via linear layers, batch normalization, ReLU activation, and dropout (0.3). Trained with TripletLoss and a dynamic margin (increasing from 1.0 to 3.0 over epochs).
  - **Training**: Focused on generating embeddings optimized for face similarity using hard and semi-hard triplet mining.

- **Ensemble**:
  - **Method**: Late fusion of embeddings from ViT and FaceNet. During inference, distances between test embeddings and reference embeddings (averaged per identity from the training set) are computed separately for each model. These distances are combined using a weighted average (FaceNet weight: 0.69, ViT weight: 0.31), and the identity with the minimum combined distance is selected, subject to a threshold (0.435) to determine if the face is "doesn't_exist".

## Results

Our team secured **3rd place** with 96% accuracy in the Fawry Kaggle competition with this ensemble approach.
