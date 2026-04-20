import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
from dvclive import Live   # ✅ FIXED
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    log_file_path = os.path.join(log_dir, 'model_evaluation.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    with open(params_path, 'r') as file:
        return yaml.safe_load(file)


def load_model(file_path: str):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def evaluate_model(clf, x_test: np.ndarray, y_test: np.ndarray):
    y_pred = clf.predict(x_test)

    if hasattr(clf, "predict_proba"):
        y_pred_proba = clf.predict_proba(x_test)[:, 1]
    else:
        y_pred_proba = y_pred

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }, y_pred   # ✅ RETURN y_pred ALSO


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save metrics to JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # ✅ Convert numpy types → Python native types
        clean_metrics = {k: float(v) for k, v in metrics.items()}

        with open(file_path, 'w') as file:
            json.dump(clean_metrics, file, indent=4)

        logger.debug('Metrics saved to %s', file_path)

    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise


def main():
    try:
        params = load_params('params.yaml')
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')

        x_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics, y_pred = evaluate_model(clf, x_test, y_test)
    
     # Experiment tracking using dvclive
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test))
            live.log_metric('precision', precision_score(y_test, y_test))
            live.log_metric('recall', recall_score(y_test, y_test))

            live.log_params(params)

        save_metrics(metrics, 'reports/metrics.json')

    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()