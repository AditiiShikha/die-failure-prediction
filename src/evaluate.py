"""Evaluation matching the company's specified grading logic (company_provided/README.md).

Metrics are computed on whatever y_true/y_pred are passed in. Callers must
pass the eligible-only (old_label==0) arrays — build_model_a_dataset()
already guarantees this by construction.
"""
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score


def evaluate_predictions(y_true, y_pred, y_proba=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    pass_recall = tn / (tn + fp) if (tn + fp) > 0 else float("nan")     # Pass Accuracy (Recall)
    fail_recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")     # Fail Accuracy (Recall)
    pass_precision = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    fail_precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    pass_f1 = (2 * pass_precision * pass_recall / (pass_precision + pass_recall)
               if (pass_precision + pass_recall) > 0 else float("nan"))
    fail_f1 = (2 * fail_precision * fail_recall / (fail_precision + fail_recall)
               if (fail_precision + fail_recall) > 0 else float("nan"))

    metrics = {
        "n_eligible": int(len(y_true)),
        "n_actual_fail": int(tp + fn),
        "n_actual_pass": int(tn + fp),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "overall_accuracy": accuracy_score(y_true, y_pred),
        "pass_accuracy_recall": pass_recall,
        "fail_accuracy_recall": fail_recall,
        "pass_precision": pass_precision,
        "fail_precision": fail_precision,
        "pass_f1": pass_f1,
        "fail_f1": fail_f1,
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    return metrics


def select_best_threshold(y_true, y_proba, thresholds=None, metric: str = "fail_f1"):
    """Sweep thresholds and return the one maximizing `metric` from evaluate_predictions.

    Caller is responsible for only passing train/CV data here (e.g. pooled
    out-of-fold probabilities) - this function has no knowledge of which
    split it's being called on, so it cannot itself prevent test/validation
    label use.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)
    best_threshold, best_score = None, -np.inf
    history = []
    for t in thresholds:
        pred = (np.asarray(y_proba) >= t).astype(int)
        score = evaluate_predictions(y_true, pred)[metric]
        history.append((float(t), score))
        if not np.isnan(score) and score > best_score:
            best_score, best_threshold = score, float(t)
    return best_threshold, best_score, history


def format_report(metrics: dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
    lines.append(f"{'':22s}{'Pred Fail':>10s} {'Pred Pass':>10s}       {'Metric':<22s}{'Value':>10s}")
    lines.append(f"{'Actual Fail':22s}{metrics['tp']:>10d} {metrics['fn']:>10d}       "
                 f"{'Fail Accuracy':<22s}{metrics['fail_accuracy_recall']:>10.6f}")
    lines.append(f"{'Actual Pass':22s}{metrics['fp']:>10d} {metrics['tn']:>10d}       "
                 f"{'Pass Accuracy':<22s}{metrics['pass_accuracy_recall']:>10.6f}")
    lines.append("")
    lines.append(f"  Overall Accuracy:  {metrics['overall_accuracy']:.6f}")
    lines.append(f"  Pass Precision:    {metrics['pass_precision']:.6f}   Fail Precision: {metrics['fail_precision']:.6f}")
    lines.append(f"  Pass F1:           {metrics['pass_f1']:.6f}   Fail F1:        {metrics['fail_f1']:.6f}")
    if "roc_auc" in metrics:
        lines.append(f"  ROC-AUC:           {metrics['roc_auc']:.6f}   PR-AUC:         {metrics['pr_auc']:.6f}")
    lines.append(f"  n_eligible={metrics['n_eligible']}  n_actual_fail={metrics['n_actual_fail']}  n_actual_pass={metrics['n_actual_pass']}")
    return "\n".join(lines)
