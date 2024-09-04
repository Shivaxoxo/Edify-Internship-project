# Cancer Cell Prediction

This repository contains the implementation of a machine learning model designed to predict cancer cells based on various input features. The project is developed as part of the Edify Internship.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
  

## Overview
The Cancer Cell Prediction project aims to accurately classify cells as cancerous or non-cancerous using machine learning techniques. This project is built using Python, leveraging popular libraries like scikit-learn and TensorFlow.

## Dataset
The dataset used for this project is publicly available and contains features extracted from cell images, such as radius, texture, perimeter, area, smoothness, and more. The target variable is a binary classification indicating whether a cell is cancerous or not.

**Dataset Source:** [https://www.kaggle.com/datasets/erdemtaha/cancer-data]()

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/shivaxoxo/Edify-Internship-Project.git
    ```

2. Navigate to the project directory:
    ```bash
    cd Edify-Internship-Project
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
After installation, you can start using the model for prediction:

1. **Preprocess the data:** Ensure the dataset is clean and split into training and testing sets.

2. **Train the model:** Run the script to train the model.
    ```bash
    python train.py
    ```

3. **Make predictions:** Use the trained model to make predictions on new data.
    ```bash
    python predict.py --input data.csv --output results.csv
    ```

## Model Details
The notebook evaluates multiple models including Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Decision Trees, and Random Forest. Each model is trained and tested on the Breast Cancer Wisconsin dataset.

Model Details:
Logistic Regression: Simple, interpretable, used as a baseline.
KNN: Non-parametric, sensitive to feature scaling.
SVM: Effective in high-dimensional spaces.
Decision Trees: Easy to interpret but prone to overfitting.
Random Forest: An ensemble method that mitigates overfitting.


## Results
Best Model: Random Forest achieved the highest accuracy (around 96%).

## Contributing
Contributions are welcome! If you would like to contribute to this project, please fork the repository and create a pull request with your changes.

