# Chest X-Ray Pneumonia Detection Using Deep Learning Models

## Project Overview

This project focuses on the automated analysis of chest X-ray images for the binary classification of two classes:

*   NORMAL
*   PNEUMONIA

The main objective is to develop and compare multiple deep learning models for pneumonia detection and to prepare their outputs for a future ensemble learning phase.

This work represents Phase 1 of the project, which includes:

*   preparing separate notebooks for each model,
*   training and evaluating the main classification models,
*   exporting prediction files for later ensemble integration.

## Dataset

The project uses a Chest X-Ray Pneumonia dataset organized into three main folders:

*   train
*   val
*   test

Each split contains two classes:

*   NORMAL
*   PNEUMONIA

Notes:

*   The dataset contains chest X-ray images for binary classification.
*   The test set was kept separate for final evaluation.
*   In some notebooks, a validation split was created from the training set because the original validation set was very small.

## Project Task

The main task in this project is:

Binary classification of chest X-ray images into NORMAL or PNEUMONIA.

The classification models included in this phase are:

*   DenseNet121
*   EfficientNetB0
*   Swin-style lightweight fallback / prototype

In addition, a U-Net notebook is included as a segmentation and preprocessing module for lung-region extraction.

## Models Included

### DenseNet121

A transfer learning classification model used as a baseline for pneumonia detection.

### EfficientNetB0

A transfer learning classification model selected for its strong performance and parameter efficiency.

### Swin Transformer (Lightweight Fallback / Prototype)

A lightweight fallback notebook prepared for the Swin experiment under Colab constraints. It follows the same dataset and output format used by the other classification notebooks.

### U-Net

A segmentation and preprocessing notebook used for lung-region extraction. This notebook is not a direct member of the classification soft-voting ensemble.

## Why the Workflow Was Unified

To prepare for future ensemble learning, the classification notebooks were aligned as much as possible in the following aspects:

*   same dataset
*   same labels
*   same test set
*   same prediction output format

This was necessary so that the outputs of the classification models could be compared fairly and combined later.

## Output Files

Each classification notebook exports a CSV file with the following columns:

*   filename
*   true\_label
*   pred\_label
*   prob\_pneumonia

Exported prediction files:

*   densenet\_test\_predictions.csv
*   efficientnet\_test\_predictions.csv
*   swin\_test\_predictions.csv

These files are intended for the next phase of the project, where model outputs can be combined using ensemble methods.

## Repository Contents

### Notebooks

*   DenseNet.ipynb
*   EfficientNet.ipynb
*   Swin\_Transformer.ipynb
*   u\_Net.ipynb

### Prediction Files

*   densenet\_test\_predictions.csv
*   efficientnet\_test\_predictions.csv
*   swin\_test\_predictions.csv

### Optional Model Files

Model weight files (.keras) may be kept separately because they are relatively large and are not always required in the repository.

Examples:

*   best\_densenet\_model.keras
*   best\_efficientnet\_model.keras
*   best\_swin\_model.keras

## Methodology Summary

The overall workflow followed in the classification notebooks is:

1.  Load and verify the dataset structure
2.  Apply preprocessing and moderate data augmentation
3.  Build the selected transfer learning model
4.  Train the model using callbacks and validation monitoring
5.  Evaluate performance on the held-out test set
6.  Export prediction probabilities for future ensemble learning

For the U-Net notebook, the workflow is:

1.  Define the segmentation architecture
2.  Prepare the segmentation and preprocessing pipeline
3.  Use lung masks or predicted masks to isolate the lung region
4.  Optionally pass the processed image to downstream classification models

## Important Project Note

The notebooks in this project do not all serve the same role:

*   DenseNet, EfficientNet, and the Swin-style notebook are classification models.
*   U-Net is a segmentation and preprocessing notebook.

Therefore, U-Net is included as a support module and not as a direct classification voting model.

## Current Phase Outcome

Phase 1 provides:

*   ready-to-run notebooks for all included models,
*   completed runs for the main classification models,
*   prediction CSV files prepared for the next ensemble phase,
*   and a structured U-Net preprocessing notebook.

## Conclusion

This project establishes a structured deep learning workflow for chest X-ray pneumonia analysis using multiple architectures. The classification notebooks provide comparable outputs for future ensemble integration, while the U-Net notebook supports lung segmentation and preprocessing as a complementary component of the overall pipeline.
