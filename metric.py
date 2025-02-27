import numpy as np
from sklearn.metrics import precision_score, recall_score, classification_report, accuracy_score
from scipy.stats import mode, skew, kurtosis, median_abs_deviation

def imbalanced_ratio(y):
    class_counts = np.bincount(y)
    return np.sum(class_counts) / np.sum(class_counts**2)

def accuracy(y, y_pred):
    return accuracy_score(y, y_pred)

def mean(x):
    return np.mean(x)

def mode(x):
    return mode(x)

def median(x):
    return np.median(x)

def min(x):
    return np.min(x)
    
def max(x):
    return np.max(x)

def range(x):
    return np.max(x) - np.min(x)

def q1(x):
    return np.percentile(x, 25)

def q2(x):
    return np.percentile(x, 50)

def q3(x):
    return np.percentile(x, 75)

def std(x):
    return np.std(x)

def var(x):
    return np.var(x)

def median_abs_deviation(x):
    return median_abs_deviation(x)

def skew(x):
    return skew(x)

def kurt(x):
    return kurtosis(x)

def pearson_skewness(x):
    return (3 * (np.mean(x) - np.median(x))) / np.std(x)