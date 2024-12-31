"""This module contains functions to evaluate the performance of a classification model."""
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix
)

def calculate_metrics(
    predictions_df: pd.DataFrame, 
    y_true_col: str , 
    y_pred_col: str
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for the specified level.
    param y_true_col: The column containing the true labels
    param y_pred_col: The column containing the predicted labels
    return: Dictionary containing accuracy, classification report, and confusion matrix
    """
    true_labels = predictions_df[y_true_col]
    predictions_df[y_pred_col] = predictions_df[y_pred_col].str.lower()
    pred_labels = predictions_df[y_pred_col]
    
    metrics = {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'classification_report': classification_report(true_labels, pred_labels),
        'confusion_matrix': confusion_matrix(true_labels, pred_labels)
    }
    return metrics

def save_evaluation_results(metrics: dict, output_path: str) -> None:
    """
    Save evaluation metrics to a text file.
    param metrics: Dictionary containing evaluation metrics
    param output_path: Path to save the results
    param output_file_name: Name of the output file
    """
    
    with open(output_path, 'w') as f:
        f.write('=' * 50 + '\n\n')
        
        f.write('Accuracy Score:\n')
        f.write(f"{metrics['accuracy']}\n\n")
        
        f.write('Classification Report:\n')
        f.write(f"{metrics['classification_report']}\n\n")
        
        f.write('Confusion Matrix:\n')
        f.write(str(metrics['confusion_matrix']))
