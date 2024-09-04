# Edify-Internship-project

#Cancer Cell Prediction
Overview
This project aims to classify cancer cells as either benign or malignant using various machine learning models. The dataset used is "Cancer_Data.csv," which contains 570 observations of cancer cells with 30 features. The classification models implemented include Logistic Regression, Decision Tree, Random Forest, and Multi-layer Perceptron.

Data Description
The dataset consists of the following columns:

id: Unique identifier for each patient
diagnosis: Target variable indicating whether the cell is benign (B) or malignant (M)
radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave points_mean, symmetry_mean, fractal_dimension_mean: Mean values of different cell properties
radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave points_se, symmetry_se, fractal_dimension_se: Standard deviations of the cell properties
radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave points_worst, symmetry_worst, fractal_dimension_worst: Worst values of cell properties
Setup
Clone the Repository

sh
Copy code
git clone <repository-url>
cd cancer-cell-prediction
Install Dependencies

Ensure you have Python installed. Install the necessary packages using:

sh
Copy code
pip install -r requirements.txt
The requirements.txt should include:

Copy code
pandas
numpy
matplotlib
seaborn
scikit-learn
Download the Dataset

Download the dataset from here and place it in the project directory.

Usage
Load and Preprocess Data

The dataset is loaded and preprocessed, including removing unnecessary columns and handling categorical data.

Train and Evaluate Models

Logistic Regression: Default and Tuned versions
Decision Tree: Default and Tuned versions
Random Forest: Default and Tuned versions
Multi-layer Perceptron
Each model is trained on the training data and evaluated on the test data. Metrics include precision, recall, F1-score, accuracy, and AUC.

Visualize Results

Visualizations include:

Distribution of benign vs. malignant cells
Histograms of continuous variables
Boxplots for each feature by diagnosis
Performance metrics for each model
Results
Logistic Regression achieved high performance with both default and tuned parameters.
Decision Tree showed good results, with improved performance after tuning.
Random Forest performed well, with the tuned model showing the best performance.
Multi-layer Perceptron also provided strong results.
The tuned Random Forest model achieved the highest AUC, while the Logistic Regression model with optimal hyperparameters had the highest balanced accuracy.

Next Steps
Compare models based on various metrics such as AUC, sensitivity, specificity, and balanced accuracy.
Interpret feature importance using the best-performing model.
Contributing
If you have suggestions or improvements, feel free to create a pull request or open an issue.


Feel free to adjust the content based on any specific details or modifications in your project.




