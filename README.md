# GSNR Prediction with Deep Learning

## Repository Overview
This repository contains implementations of deep learning models used for **GSNR (Generalized Signal-to-Noise Ratio)** prediction. The research was conducted at the **Optical Networks and Technologies Lab** in Islamabad, Pakistan, from **May 2024 to August 2024**. The primary goal was to develop highly efficient models to predict GSNR values while minimizing the **Mean Squared Error (MSE)**, achieving an accuracy of less than **0.01 MSE units**.

### Key Highlights:
- **Project Duration**: May 2024 - August 2024
- **Research Lab**: Optical Networks and Technologies Lab, Islamabad
- **Objective**: Predict GSNR values using deep learning models with enhanced accuracy.
- **Techniques**: Transfer Learning, Active Learning, Federated Learning, and Knowledge Distillation.
- **Explainable AI (XAI)**: Interpretation of model predictions using SHAP, LIME, ICE, and PDP plots.

## Table of Contents
1. [Model Implementations](#model-implementations)
   - [T1: EuTopology_GSNRPrediction.ipynb](#t1-eutopology_gsnrpredictionipynb)
   - [T2: Neural Network](#t2-neuralnetipynb)
   - [T3: Transfer Learning](#t3-transferlearningipynb)
   - [T4: Active Learning](#t4-activelearningipynb)
   - [T5: Knowledge Distillation](#t5-knowledgedistillationipynb)
   - [T7: Explainable AI](#t7-explainableaiipynb)
2. [Performance Evaluation](#performance-evaluation)
3. [Installation](#installation)

## Model Implementations

### T1: EuTopology_GSNRPrediction.ipynb
- **Description**: This notebook focuses on the prediction of GSNR values for an optical network using Random Forest and other regression models.
- **Key Feature**: Utilizes Random Forest regression with hyperparameter tuning to achieve high prediction accuracy.

### T2: NeuralNet.ipynb
- **Description**: Implementation of a basic neural network model to predict GSNR values. Initial steps in improving prediction with deep learning techniques.
- **Key Feature**: Comparison with traditional machine learning models.

### T3: TransferLearning.ipynb
- **Description**: Uses transfer learning to leverage pre-trained models for GSNR prediction, which enhances the learning process and prediction accuracy.
- **Key Feature**: Efficient reuse of pre-trained knowledge to improve model performance.

### T4: ActiveLearning.ipynb
- **Description**: Incorporates active learning techniques to fine-tune the GSNR prediction model, optimizing the model by selecting the most informative samples.
- **Key Feature**: Improves model performance by strategically selecting data points.

### T5: KnowledgeDistillation.ipynb
- **Description**: Implements knowledge distillation to transfer knowledge from a large, complex model (teacher) to a smaller, simpler model (student).
- **Key Feature**: Significant reduction in model complexity while retaining high accuracy.

### T7: ExplainableAI.ipynb
- **Description**: The results of GSNR prediction models are explained using various Explainable AI techniques such as SHAP, LIME, ICE, and PDP.
- **Key Feature**: Interpretation of model behavior and feature importance using XAI.

### T7USAExplainableAI.ipynb
- **Description**: The results of GSNR prediction models are explained using various Explainable AI techniques such as SHAP, LIME, ICE, and PDP on a larger USA Network Topology Dataset
- **Key Feature**: Interpretation of model behavior and feature importance using XAI.

## Performance Evaluation
The GSNR prediction models were evaluated based on their Mean Squared Error (MSE) and explained using XAI techniques to understand feature contributions. The Random Forest model and gradient boosting models achieved an MSE of **<0.01 units** along with a custom-made ANNs, which is a significant achievement for this problem.

Explainable AI techniques like **SHAP**, **LIME**, **ICE**, and **PDP** were used to visualize the contributions of each feature in the prediction, providing clarity on the model’s decision-making process.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/saadan1234/Optical-Network-and-Technologies.git
