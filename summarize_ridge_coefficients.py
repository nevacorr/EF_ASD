import numpy as np
import pandas as pd


def summarize_ridge_coefficients(coefs_bootstrap, predictor_names, top_n=10):

    mean_coefs = np.mean(coefs_bootstrap, axis=0)
    std_coefs = np.std(coefs_bootstrap, axis=0)
    abs_mean_coefs = np.abs(mean_coefs)
    positive_counts = np.sum(coefs_bootstrap > 0, axis=0)
    negative_counts = np.sum(coefs_bootstrap < 0, axis=0)
    num_bootstraps = coefs_bootstrap.shape[0]

    coef_summary = pd.DataFrame({
        'Predictor': predictor_names,
        'MeanCoefficient': mean_coefs,
        'StdCoefficient': std_coefs,
        'AbsMeanCoefficient': abs_mean_coefs,
        'PositiveFraction': positive_counts / num_bootstraps,
        'NegativeFraction': negative_counts / num_bootstraps
    })

    coef_summary = coef_summary.sort_values(by='AbsMeanCoefficient', ascending=False)

    return coef_summary