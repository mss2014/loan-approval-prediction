# loan-approval-prediction
A Python project demonstrating the prediction of loan approval status using machine learning. Compares the performance of Decision Tree, K-Nearest Neighbors (KNN), and Support Vector Classifier (SVC) algorithms on the loan_approval_dataset.csv. Includes data preprocessing, model training, evaluation, and comparison.
# Loan Approval Prediction using Classification Algorithms

## Objective

This project aims to build and evaluate machine learning models to predict loan approval based on applicant data from the `loan_approval_dataset.csv`. The primary goal is to compare the performance of three different classification algorithms: Decision Tree (DT), K-Nearest Neighbors (KNN), and Support Vector Classifier (SVC), using accuracy and other classification metrics.

## Dataset

*   **`loan_approval_dataset.csv`**: This file contains the raw data used for the analysis. It includes various features of loan applicants such as income, number of dependents, education level, employment status, CIBIL score, asset values, loan details, and the final loan approval status (`loan status`).

## Methodology

The analysis involved the following steps:

1.  **Data Loading:** The dataset was loaded using the Pandas library.
2.  **Data Preprocessing:**
    *   **Column Cleaning:** Removed leading/trailing whitespace from column names.
    *   **Handling Invalid Values:** Identified and handled illogical negative values in asset columns by replacing them with 0.
    *   **Feature Dropping:** Removed the `loan_id` column as it's an identifier.
    *   **Label Encoding:** Converted categorical features (`education`, `self_employed`, `loan_status`) into numerical format using `LabelEncoder`.
    *   **Data Splitting:** Split the data into training (80%) and testing (20%) sets, stratifying by the `loan_status` target variable.
3.  **Model Implementation:**
    *   Implemented three classification algorithms from Scikit-learn:
        *   Decision Tree Classifier (`DecisionTreeClassifier`)
        *   K-Nearest Neighbors Classifier (`KNeighborsClassifier` with k=5)
        *   Support Vector Classifier (`SVC` with default RBF kernel)
4.  **Model Training:** Trained each classifier on the scaled training data.
5.  **Model Evaluation:**
    *   Evaluated each trained model on the scaled testing data.
    *   Calculated and reported:
        *   Accuracy Score
        *   Confusion Matrix 
        *   Classification Report (including precision, recall, F1-score)
6.  **Comparison:** Compiled the accuracy results into a comparison table.

## Files in this Project

*   **`loan_approval_dataset.csv`**: The input dataset containing applicant information and loan status.
*   **`loan_approval_prediction.ipynb`**: The Python script containing all the code for data preprocessing, model training, evaluation, and result generation.
*   **`Loan Approval Classification Report.pdf`** : The final report document detailing the analysis, results, and conclusions.
*   **`README.md`**: This file, providing an overview of the project.

## Technologies Used

*   Python 3.x
*   Pandas: For data manipulation and loading CSV files.
*   NumPy: For numerical operations.
*   Scikit-learn: For machine learning tasks including preprocessing (LabelEncoder, StandardScaler, train_test_split), modeling (DecisionTreeClassifier, KNeighborsClassifier, SVC), and evaluation (accuracy_score, confusion_matrix, classification_report).

## Setup and Usage

1.  Ensure you have Python 3 installed.
2.  Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
    *(Alternatively, if a `requirements.txt` file is provided: `pip install -r requirements.txt`)*
3.  Place the `loan_approval_dataset.csv` file in the same directory as the script.
4.  Run the Python script:
    ```bash
    python loan_approval_prediction.ipynb
    ```
5.  The script will output the evaluation metrics (accuracy, confusion matrix, classification report) for each model to the console and display the confusion matrix plots. The comparison table will also be printed. Refer to the `Loan Approval Prediction Analysis Report.pdf` (or `.docx`) for a detailed discussion.

## Results Summary

The performance of the three classifiers was compared based on accuracy on the test set:

| Classifier                |   Accuracy (%) |
|:--------------------------|---------------:|
| Decision Tree             |          97.19 |
| K-Nearest Neighbors       |          89.70 |
| Support Vector Classifier |          94.26 |

The Decision Tree Classifier achieved the highest accuracy, slightly outperforming SVC and significantly outperforming KNN for this specific dataset and preprocessing pipeline.
