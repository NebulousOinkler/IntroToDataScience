# metrics.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from tabulate import tabulate  # To pretty-print the data

# Data Cleaning Section

def clean_data(df):
    """
    Clean the medical data (e.g., Diabetes diagnosis) by:
    1. Handling missing values
    2. Encoding categorical features (if any)
    3. Normalizing or scaling numeric features (if necessary)
    
    :param df: pandas DataFrame containing raw data
    :return: Cleaned pandas DataFrame
    """
    
    df_cleaned = df.copy()
    
    # 1. Handle Missing Values (Imputation)
    # We'll use SimpleImputer to fill missing values. For numerical columns, we use mean imputation.
    numeric_cols = df_cleaned.select_dtypes(include=np.number).columns
    imputer = SimpleImputer(strategy='mean')
    df_cleaned[numeric_cols] = imputer.fit_transform(df_cleaned[numeric_cols])
    
    # 2. Encode Categorical Features (if any)
    # Example: If the dataset contains a 'Gender' column (categorical), we'll encode it
    categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()
    
    for col in categorical_columns:
        df_cleaned[col] = label_encoder.fit_transform(df_cleaned[col])
    
    # If there are any non-numeric columns that aren't relevant for the model, drop them
    # Example: Drop patient IDs or other columns not used for the prediction task
    df_cleaned.drop(['Patient_ID'], axis=1, inplace=True, errors='ignore')
    
    return df_cleaned

# Function to calculate confusion matrix based on true and predicted labels
def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix based on true and predicted labels.

    :param y_true: List of true labels (0 or 1)
    :param y_pred: List of predicted labels (0 or 1)
    :return: Confusion matrix (TP, FP, TN, FN)
    """
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    return tp, fp, tn, fn

# Precision: the proportion of positive predictions that are actually correct
def precision(tp, fp):
    """
    Calculate precision based on confusion matrix values.

    :param tp: True Positives
    :param fp: False Positives
    :return: Precision score
    """
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

# Recall (True Positive Rate): the proportion of actual positives that are correctly identified
def recall(tp, fn):
    """
    Calculate recall based on confusion matrix values.

    :param tp: True Positives
    :param fn: False Negatives
    :return: Recall score
    """
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

# Accuracy: the proportion of total correct predictions
def accuracy(tp, tn, total_instances):
    """
    Calculate accuracy based on confusion matrix values.

    :param tp: True Positives
    :param tn: True Negatives
    :param total_instances: Total number of instances
    :return: Accuracy score
    """
    return (tp + tn) / total_instances

# Specificity: the proportion of actual negatives correctly identified
def specificity(tn, fp):
    """
    Calculate specificity based on confusion matrix values.

    :param tn: True Negatives
    :param fp: False Positives
    :return: Specificity score
    """
    if tn + fp == 0:
        return 0.0
    return tn / (tn + fp)

# F1 Score: the harmonic mean of precision and recall
def f1_score(precision, recall):
    """
    Calculate F1 score based on precision and recall.

    :param precision: Precision score
    :param recall: Recall score
    :return: F1 score
    """
    if precision + recall == 0:
        return 0.0
    return 2 / ((1 / precision) + (1 / recall))

# ROC curve and AUC calculation
def roc_curve_and_auc(y_true, y_scores):
    """
    Generate ROC curve and calculate AUC using pure Python.

    :param y_true: List of true binary labels (0 or 1)
    :param y_scores: List of predicted scores (probabilities, not hard classification labels)
    :return: False Positive Rate, True Positive Rate, AUC score
    """
    # Sort by score and threshold values
    thresholds = sorted(set(y_scores), reverse=True)
    
    # Variables to store the True Positive Rate (TPR) and False Positive Rate (FPR)
    tpr = []
    fpr = []

    for threshold in thresholds:
        # Predicted labels based on the threshold
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        
        # Calculate confusion matrix for each threshold
        tp, fp, tn, fn = calculate_confusion_matrix(y_true, y_pred)

        # Calculate True Positive Rate (Recall) and False Positive Rate
        tpr.append(recall(tp, fn))  # TPR = Recall
        fpr.append(fp / (fp + tn))  # FPR = FP / (FP + TN)

    # Calculate AUC using the trapezoidal rule
    auc_score = np.trapezoid(tpr, fpr)  # Numerical integration using numpy.trapz()

    # Plotting the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    return fpr, tpr, auc_score

def auc_score(fpr, tpr):
    """
    Calculate the Area Under the Curve (AUC) using the trapezoidal rule.

    :param fpr: False Positive Rates from ROC curve
    :param tpr: True Positive Rates from ROC curve
    :return: AUC score
    """
    return np.trapezoid(tpr, fpr)

# Pretty print the DataFrame for initial view
def pretty_print_df(df):
    """
    Print the raw DataFrame in a clean tabular format before any processing.
    """
    print("\n--- Raw Dataset (First 10 Entries) ---")
    print(tabulate(df.head(10), headers='keys', tablefmt='pretty', showindex=False))  # Display first 10 rows

# Pretty print the metrics results
def pretty_print_metrics(precision, recall, accuracy, specificity, f1, auc):
    """
    Print the model performance metrics in a clean format.
    """
    print("\n--- Model Performance Metrics ---")
    metrics = {
        'Precision': precision,
        'Recall': recall,
        'Accuracy': accuracy,
        'Specificity': specificity,
        'F1 Score': f1,
        'AUC': auc
    }
    
    print(tabulate(metrics.items(), headers=['Metric', 'Value'], tablefmt='pretty'))

# Example Usage: Simulated Data for Diabetes Diagnosis
if __name__ == "__main__":
    # Generate a larger dataset of 1000 patients
    np.random.seed(42)  # For reproducibility

    # Randomly generate features for the dataset
    num_samples = 1000
    patient_ids = np.arange(1, num_samples + 1)
    ages = np.random.randint(18, 90, size=num_samples)  # Random ages between 18 and 90
    genders = np.random.choice(['Male', 'Female'], size=num_samples)  # Random genders
    glucose_levels = np.random.randint(70, 200, size=num_samples)  # Random glucose levels
    has_disease = np.random.choice([0, 1], size=num_samples)  # Random disease label: 1 = Has diabetes, 0 = No diabetes

        # Create DataFrame from the generated data
    df = pd.DataFrame({
        'Patient_ID': patient_ids,
        'Age': ages,
        'Gender': genders,
        'Glucose_Level': glucose_levels,
        'Has_Disease': has_disease
    })

    # Pretty-print the raw data before processing
    pretty_print_df(df)

    # Clean the data
    df_cleaned = clean_data(df)
    pretty_print_df(df_cleaned)
    # Now we simulate more realistic predictions
    # Simulating predicted probabilities based on 'Glucose_Level' and other features
    # Assume that higher glucose levels correlate with a higher likelihood of having the disease
    glucose_scores = df_cleaned['Glucose_Level'].values
    # Simulating probabilities: Higher glucose levels increase probability of disease (logistic-like model)
    predicted_probabilities = np.clip(1 / (1 + np.exp(-0.05 * (glucose_scores - 100))), 0, 1)

    # TODO. Make a new model and try to get better results! 


    # Threshold to convert probabilities to binary predictions (0 or 1)
    y_pred = (predicted_probabilities >= 0.5).astype(int)  # Using 0.5 as the threshold

    # True labels
    y_true = df['Has_Disease'].values

    # Calculate confusion matrix and metrics
    tp, fp, tn, fn = calculate_confusion_matrix(y_true, y_pred)
    precision_value = precision(tp, fp)
    recall_value = recall(tp, fn)
    accuracy_value = accuracy(tp, tn, len(y_true))
    specificity_value = specificity(tn, fp)
    f1_value = f1_score(precision_value, recall_value)

    # Calculate AUC (using simulated predicted probabilities)
    fpr, tpr, auc_value = roc_curve_and_auc(y_true, predicted_probabilities)

    # Pretty-print the metrics results after model evaluation
    pretty_print_metrics(precision_value, recall_value, accuracy_value, specificity_value, f1_value, auc_value)