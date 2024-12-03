import csv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

def save_data(model,X_test,y_test,name):
        
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Predict probabilities and classes
    y_pred_probs = model.predict(X_test)  # Predicted probabilities
    y_pred_classes = y_pred_probs.argmax(axis=1)  # Predicted class labels
    
    # Calculate precision, recall, and f1 score for each class
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average=None)  # Recall for each class
    f1 = f1_score(y_test, y_pred_classes, average='weighted')
    
    # One-hot encode y_test for AUC calculation
    num_classes = len(set(y_test))  # Adjust according to your dataset
    y_test_binarized = label_binarize(y_test, classes=range(num_classes))
    
    # Compute AUC for each class
    auc_per_class = roc_auc_score(y_test_binarized, y_pred_probs, average=None)
    
    # Prepare data to save
    model_name = name
    data = {
        "Model Name": model_name,
        "Loss": loss,
        "Accuracy": accuracy,
        "Precision": precision,
        "F1 Score": f1
    }
    
    # Add Recall for each class
    for i, recall_value in enumerate(recall):
        data[f"Recall_Class_{i}"] = recall_value
    
    # Add AUC values for each class
    for i, auc_value in enumerate(auc_per_class):
        data[f"AUC_Class_{i}"] = auc_value
    
    # Save to CSV
    csv_file = "model_metrics.csv"
    file_exists = False
    
    try:
        with open(csv_file, mode='r') as f:
            file_exists = True
    except FileNotFoundError:
        pass
    
    # Append or write new file
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        
        # Write header only if the file is new
        if not file_exists:
            writer.writeheader()
        
        # Write data row
        writer.writerow(data)
    
    print(f"Metrics saved to {csv_file}")
