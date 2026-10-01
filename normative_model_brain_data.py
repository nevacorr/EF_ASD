import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from covbat_harmonize_normative_group import covbat_harmonize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
from helper_functions_clustering import plot_brain_vs_age_by_sex_from_model

def calc_normative_data(df, group_col='Group', lr_label='LR-', hr_labels=['HR+', 'HR-'],
                        brain_cols=None, covariates=['Sex', 'Final_Age_School_Age'], random_state=42):
    """
    1. Fit normative models on LR kids
    2. Compute z-scores for HR kids
    """
    # --------------- 1. Split dataframe ---------------
    df_lr_noth = df[df[group_col] == lr_label].copy()
    df_hr_noth = df[df[group_col].isin(hr_labels)].copy()
    df_nan = df[df[group_col].isna()].copy()
    df_lr_pos = df[df[group_col] == "LR+"].copy()
    print(f'df size = {df.shape[0]}, sum={df_lr_noth.shape[0]+ df_lr_pos.shape[0]+df_hr_noth.shape[0]+df_nan.shape[0]}')
    df_lr_noth.reset_index(drop=True, inplace=True)
    df_hr_noth.reset_index(drop=True, inplace=True)

    # ----------------3. Harmonize data -----------------
    df_lr_for_harmonize = df_lr_noth[['Site'] + brain_cols].copy()
    df_hr_for_harmonize = df_hr_noth[['Site'] + brain_cols].copy()
    columns_removed = [col for col in df_lr_noth.columns if col not in df_lr_for_harmonize]
    df_lr_columns_removed = df_lr_noth[columns_removed].copy()
    df_hr_columns_removed = df_hr_noth[columns_removed].copy()

    df_lr_harmonized, df_hr_harmonized = covbat_harmonize(df_lr_for_harmonize, df_hr_for_harmonize)
    df_lr = df_lr_harmonized + df_lr_columns_removed
    df_hr = df_hr_harmonized + df_hr_columns_removed

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

        # Calculate BIC  #
        residuals_linear  = y_lr - model.predict(X_lr)
        n=len(y_lr)
        k_linear = X_lr.shape[1] + 1
        bic_linear = n * np.log(np.sum(residuals_linear**2) / n) + k_linear * np.log(n)

        #Quadratic model
        X_lr_quad = np.column_stack([df_lr_clean['Sex'].values, df_lr_clean['Final_Age_School_Age'].values,
                                     df_lr_clean['Final_Age_School_Age'].values**2])
        model_quad = LinearRegression()
        model_quad.fit(X_lr_quad, y_lr)
        residuals_quad = y_lr - model_quad.predict(X_lr_quad)
        k_quad = X_lr_quad.shape[1] + 1
        bic_quad = n * np.log(np.sum(residuals_quad**2) / n) + k_quad * np.log(n)
        rmse_linear = np.sqrt(np.mean(residuals_linear ** 2))
        rmse_quad = np.sqrt(np.mean(residuals_quad ** 2))
        print(f'{col} bic_linear = {bic_linear}, bic_quad = {bic_quad}, '
              f'rmse_linear = {rmse_linear} rmse_quad = {rmse_quad}')

        # plot_brain_vs_age_by_sex_from_model(X_lr, y_lr, col, model)

        # Predicted for HR kids
        df_hr_clean = df_hr.dropna(subset=covariates + [col]).copy()
        X_hr = df_hr_clean[covariates].values
        y_pred_hr = model.predict(X_hr)
        y_actual_hr = df_hr_clean[col].values

        # SD of residuals in LR
        resid_std = np.std(residuals_linear, ddof=X_lr.shape[1] + 1)
        print(f'resid sd {col} = {resid_std}')

        # Z-score for HR kids
        z_col = f"{col}_z"
        z_cols.append(z_col)
        df_tmp = pd.DataFrame({'CandID': df_hr_clean['CandID'], z_col:(y_actual_hr - y_pred_hr) / resid_std})
        df_tmp.reset_index(inplace=True, drop=True)
        df_hr_z = df_hr_z.merge(df_tmp, on='CandID', how='left')

        # plot_brain_vs_age_by_sex_from_model(X_hr, y_actual_hr, col, model)

    # Drop HR rows with any NaNs in z-scores
    df_hr_z= df_hr_z.dropna(subset=z_cols, how='any')

    # reindex rows
    df_hr_z.reset_index(inplace=True, drop=True)

    return df_hr_z
