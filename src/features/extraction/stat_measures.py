import numpy as np
from scipy.stats import skew, kurtosis


class StatisticalMeasures:
    """
    A class dedicated to computing various statistical measures on a given feature array.
    This utility class provides a method to compute a predefined set of statistical measures
    which are commonly used in data analysis and feature extraction.

    Methods:
    --------
    compute_statistical_measures(feature_array, measures=None):
        Computes selected statistical measures from a provided numerical array.
    """
    @staticmethod
    def compute_statistical_measures(feature_array, measures=None):
        
        if measures is None:
            measures = ['mean', 'std', 'var', 'min', 'max', 'range', '25th_percentile', '50th_percentile', '75th_percentile', 'skew', 'kurtosis']
        
        stats = {}
        if 'mean' in measures:
            stats['mean'] = np.mean(feature_array)
        if 'std' in measures:
            stats['std'] = np.std(feature_array)
        if 'var' in measures:
            stats['var'] = np.var(feature_array)
        if 'min' in measures:
            stats['min'] = np.min(feature_array)
        if 'max' in measures:
            stats['max'] = np.max(feature_array)
        if 'range' in measures:
            stats['range'] = np.ptp(feature_array)
        if '25th_percentile' in measures:
            stats['25th_percentile'] = np.percentile(feature_array, 25)
        if '50th_percentile' in measures:
            stats['50th_percentile'] = np.percentile(feature_array, 50)
        if '75th_percentile' in measures:
            stats['75th_percentile'] = np.percentile(feature_array, 75)
        if 'skew' in measures and len(np.unique(feature_array)) > 1:
            stats['skew'] = skew(feature_array)
        else:
            stats['skew'] = np.nan
        if 'kurtosis' in measures and len(np.unique(feature_array)) > 1:
            stats['kurtosis'] = kurtosis(feature_array)
        else:
            stats['kurtosis'] = np.nan
        return stats
