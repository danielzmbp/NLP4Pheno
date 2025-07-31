"""
XGBoost-based phenotype prediction using protein domain features.

This script trains binary XGBoost classifiers to predict bacterial phenotypes
based on InterProScan protein domain annotations. For each relationship type
(e.g., STRAIN-PHENOTYPE:PRESENTS), it trains a model to distinguish between
strains that have vs. don't have specific phenotypic traits.

The pipeline:
1. Loads protein domain feature matrices from genome annotations
2. For each phenotype, creates binary classification tasks (target vs. other)
3. Trains XGBoost models with GPU acceleration
4. Evaluates model performance and extracts feature importances
5. Identifies key protein domains associated with phenotypes

This integrates the NLP-extracted relationships with genomic features to
enable phenotype prediction from genome sequences.
"""

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

# Snakemake parameters
data = snakemake.params.data  # Dataset identifier
device = snakemake.params.device[0]  # GPU device ID
path = snakemake.params.path  # Base path for data

# Load InterPro domain names for human-readable feature names
ip_names = pd.read_csv(
    "https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/entry.list",
    sep="\t",
    header=0,
)
ip_names.set_index("ENTRY_AC", inplace=True)

# Clean domain names to avoid issues with special characters
ip_names["ENTRY_NAME"] = (
    ip_names["ENTRY_NAME"]
    .str.replace("[", "_")
    .str.replace("]", "_")
    .str.replace("<", "_")
)


def calculate_acc(dtest, bst, enc, y_test_binary):
    """
    Calculate accuracy for binary XGBoost predictions.
    
    Args:
        dtest: XGBoost DMatrix containing test features
        bst: Trained XGBoost booster model
        enc: LabelEncoder for converting labels to binary format
        y_test_binary: Original test labels in string format
        
    Returns:
        float: Accuracy score between 0 and 1
    """
    # Get predicted probabilities and round to binary predictions
    y_pred_binary = bst.predict(dtest)
    predictions = [round(value) for value in y_pred_binary]
    # Calculate accuracy against encoded true labels
    acc = accuracy_score(enc.transform(y_test_binary), predictions)
    return acc


def process_rel(filepath, device, ip_names):
    """
    Process a relationship type and train XGBoost models for each phenotype.
    
    For each unique phenotype in the relationship data, trains a binary classifier
    to distinguish strains with that phenotype from others. Uses protein domain
    features as input and GPU acceleration for training.
    
    Args:
        filepath: Path to pickle file containing (X_features, y_labels, feature_indices)
        device: CUDA device ID for GPU training
        ip_names: DataFrame mapping InterPro IDs to human-readable names
        
    Returns:
        list: Tuples of (phenotype, accuracy, trained_model) for each phenotype
    """
    d_rel = []
    # Load preprocessed data: feature matrix, labels, and feature indices
    with open(filepath, "rb") as f:
        dat = pickle.load(f)
    vc = pd.DataFrame(dat[1]).value_counts()  # Count occurrences of each phenotype
    X = dat[0]  # Protein domain feature matrix
    y = dat[1]  # Phenotype labels
    ind = dat[2]  # InterPro domain indices
    
    # Convert InterPro IDs to human-readable domain names
    ind_names = [ip_names[ip_names.index == i]["ENTRY_NAME"].values[0] for i in ind]

    # Train a binary classifier for each phenotype
    for i in vc.index:
        # Create binary labels: target phenotype vs all others
        y_binary = ["target" if label == i[0] else "other" for label in y]

        # Split data maintaining class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, stratify=y, random_state=42
        )

        # Encode string labels to binary (0/1)
        enc = LabelEncoder().fit(y_train)

        # Create XGBoost data matrices with feature names
        dtrain = xgb.DMatrix(
            X_train, label=enc.transform(y_train), feature_names=ind_names
        )
        dtest = xgb.DMatrix(
            X_test, label=enc.transform(y_test), feature_names=ind_names
        )

        # XGBoost parameters optimized for phenotype prediction
        param = {
            "max_depth": 6,  # Tree depth to prevent overfitting
            "eta": 0.3,  # Learning rate
            "objective": "binary:logistic",  # Binary classification
            "device": f"cuda:{device}",  # GPU acceleration
            "eval_metric": ["logloss"],  # Log loss for binary classification
            "colsample_bylevel": 1,  # Use all features per tree level
            "booster": "gbtree",  # Tree-based model
        }

        # Train with early stopping to prevent overfitting
        evallist = [(dtrain, "train"), (dtest, "eval")]
        bst = xgb.train(
            param,
            dtrain,
            5000,  # Max rounds
            evals=evallist,
            early_stopping_rounds=10,  # Stop if no improvement for 10 rounds
            verbose_eval=False,
        )

        # Evaluate model performance
        accuracy = calculate_acc(dtest, bst, enc, y_test)
        d_rel.append([i, accuracy, bst])
        print(f"{filepath.split("/")[-1]} {i} {accuracy} {vc[i]}")
    return d_rel

# Main execution
print(f"Processing: {snakemake.input}")
result = process_rel(str(snakemake.input), device, ip_names)

# Save trained models and results
with open(snakemake.output[0], "wb") as f:
    pickle.dump(result, f)
    