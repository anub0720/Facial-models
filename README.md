# Facial-models
This contains source code,trained models and testing codes for COMSYS-2025 hackathon for team OpticEdge.


Task-A folder contains the source code which is used to obtain the checkpoint. 
To execute testing code u need to change the file_path and checkpoint path inside it. Details are given inside the notebook. 

Same for Task-B. The only difference is that instead of a checkpoint, a path file has been used. 

USAGE OF KAGGLE IS HIGHLY REQUESTED AS THE CODE HAS BEEN PRODUCED IN THAT ENVIRONMENT

In case you wish to run source codes, please use the following datasets : 

Task-A : 

Raw data : https://www.kaggle.com/datasets/anubhabbhattacharya7/comsys-taska/data

The following datasets are downloaded on running the source code. However, these are provided for easy access : 

Preprocessed data : https://www.kaggle.com/datasets/sohammandal001/new-preprocessed-images-task-a-comsys-25k/data

Face detection model files : 

https://www.kaggle.com/datasets/sohammandal001/face-detection-model-files/data

Task-B : 

Raw data : https://www.kaggle.com/datasets/anubhabbhattacharya7/comsys/data

For any queries, please contact : 
Soham Mandal (sohammandal8122005@gmail.com)

Task A : 

# Architecture Overview

The model's design can be broken down into the following key components:

---

## 1. Dual Feature Extractors (EfficientNetB3)

The system begins by processing input images through two parallel and independent **EfficientNetB3** backbone networks:

- **`self.feature_extractor_original`**:  
  Dedicated to extracting features from the original, raw images. This allows the network to capture general appearance and contextual information.

- **`self.feature_extractor_processed`**:  
  Dedicated to extracting features from the preprocessed images (e.g., face-aligned, cropped). This enables the network to specialize in learning fine-grained facial details.

Both **EfficientNetB3** instances are initialized with **pre-trained weights from ImageNet** (`pretrained=True`) and have their final classification heads removed (`num_classes=0`). Each backbone outputs a **1536-dimensional feature vector**.

This **dual-stream approach** ensures that the model can effectively utilize information from both data representations.

---

## 2. Feature Fusion and Re-calibration (SE Block)

After features are extracted from both streams, they are concatenated along the feature dimension:

features_combined = torch.cat((features_original, features_processed), dim=1)


Task B : 

# Model Architecture

The system's architecture is built upon two core components:

## 1. EmbeddingNet

The **EmbeddingNet** serves as the primary feature extractor. It utilizes a pre-trained **EfficientNet-B4** model. EfficientNet-B4 is a state-of-the-art convolutional neural network known for its excellent performance and efficiency, achieved through a compound scaling method.

We replace the original classification head of EfficientNet-B4 with a custom linear layer.

This new layer projects the extracted features into a **512-dimensional embedding space**, which represents the unique face embedding.

Using pre-trained weights from ImageNet significantly accelerates training and improves performance, especially with limited face data.

The EmbeddingNet also handles the necessary image transformations, ensuring consistent preprocessing.

## 2. ArcMarginProduct (ArcFace)

The **ArcMarginProduct** layer, commonly referred to as **ArcFace**, is a specialized loss function designed to enhance the discriminative power of the learned embeddings.

Unlike traditional softmax, ArcFace introduces an angular margin penalty and a scaling factor to the cosine similarity between the embedding and the class centers.

This mechanism forces embeddings from the same class to cluster more tightly while pushing embeddings from different classes further apart in an angular space.

During the forward pass, it normalizes the embeddings and class weights, then calculates cosine similarity.

An **additive angular margin (m)** is applied to the true class's angle, increasing the decision boundary for correct classification.

A **scaling factor (s)** further magnifies these differences, aiding in faster convergence and more distinct feature learning.

