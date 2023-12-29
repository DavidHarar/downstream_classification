import numpy as np
import pandas as pd

def PRAUC(y_true, y_pred):
    """
    Get area under the precision recall curve.
    """
    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

def ROCAUC(y_true, y_pred):
    """
    Get area under the ROC.
    """
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)

def MaxRecall_for_MinPrecision(y_true, y_pred, min_precision):
    """
    Given a precision threshold, get max value of recall.
    """
    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    closest_index = np.argmin(np.abs(precision - min_precision))
    
    return recall[closest_index], thresholds[closest_index]

def get_tpr_for_fpr_budget(y_true, y_pred, fpr_budget = 0.6):
    """
    This funciton tells for a budget of false positive rate, what rate of true positives we can get.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    closest_index = np.argmin(np.abs(fpr - fpr_budget))
    return tpr[closest_index]


def get_threshold_for_fpr_budget(y_true, y_pred, fpr_budget = 0.6):
    """
    This funciton tells for a budget of false positive rate, what rate of true positives we can get.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    closest_index = np.argmin(np.abs(fpr - fpr_budget))
    return thresholds[closest_index]

