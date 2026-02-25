import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
from helper_functions_clustering import plot_brain_vs_age_by_sex_from_model

def calc_normative_data(df, group_col='Group', lr_label='LR-', hr_labels=['HR+', 'HR-'],
                        brain_cols=None, covariates=['Sex', 'Final_Age_School_Age', 'ICV_vol_VSA'], random_state=42):
    """
    1. Fit normative models on LR kids
    2. Compute z-scores for HR kids
    """
    # --------------- 1. Split dataframe ---------------
    df_lr = df[df[group_col] == lr_label].copy()
    df_hr = df[df[group_col].isin(hr_labels)].copy()
    df_lr.reset_index(drop=True, inplace=True)
    df_hr.reset_index(drop=True, inplace=True)

    # --------------- 2. Compute z-scores ---------------
    z_cols = []
    df_hr_z = df_hr[['CandID', 'Identifiers']].copy()
    for col in brain_cols:
        # Keep only rows without NaN in covariates or the brain metric
        df_lr_clean = df_lr.dropna(subset=covariates + [col]).copy()
        # Fit linear model: brain ~ covariates in LR kids
        X_lr = df_lr_clean[covariates].values
        y_lr = df_lr_clean[col].values
        model = LinearRegression()
        model.fit(X_lr, y_lr)

        # plot_brain_vs_age_by_sex_from_model(X_lr, y_lr, col, model)

        # Predicted for HR kids
        df_hr_clean = df_hr.dropna(subset=covariates + [col]).copy()
        X_hr = df_hr_clean[covariates].values
        y_pred_hr = model.predict(X_hr)
        y_actual_hr = df_hr_clean[col].values

        # SD of residuals in LR
        resid_std = np.std(y_lr - model.predict(X_lr))

        # Z-score for HR kids

        z_col = f"{col}_z"
        z_cols.append(z_col)
        df_tmp = pd.DataFrame({'CandID': df_hr_clean['CandID'], z_col:(y_actual_hr - y_pred_hr) / resid_std})
        df_tmp.reset_index(inplace=True, drop=True)
        df_hr_z = df_hr_z.merge(df_tmp, on='CandID', how='left')

        mystop=1

    # Drop HR rows with any NaNs in z-scores
    df_hr_z= df_hr_z.dropna(subset=z_cols, how='any')

    # reindex rows
    df_hr_z.reset_index(inplace=True, drop=True)

    return df_hr_z
