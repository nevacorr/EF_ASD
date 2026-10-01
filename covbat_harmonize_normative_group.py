import os
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from Utility_Functions_XGBoost import impute_by_site_median_with_nan_indices
import rpy2.robjects as ro
os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
robjects.r('source("~/R_Projects/IBIS_EF_xgboost/covbat_wrapper_for_use_with_python.R")')

def covbat_harmonize(X_LR, X_HR):
    
    # Assign covbat functions
    fit_covbat = robjects.r['fit_covbat']
    apply_covbat = robjects.r['apply_covbat']
    
    #  Replace NaN values with column median for harmonization
    fcols = X_LR.columns.difference(['Site'])
    
    (X_LR_temp, X_HR_temp, nan_indices_train, nan_indices_test) = impute_by_site_median_with_nan_indices(
        X_LR,
        X_HR,
        feature_cols=fcols,
        site_col='Site'
    )
    
    # --- Convert to R data frames ---
    with localconverter(robjects.default_converter + pandas2ri.converter):
        X_LR_r = robjects.conversion.py2rpy(X_LR_temp.drop(columns=['Site']))
        X_HR_r = robjects.conversion.py2rpy(X_HR_temp.drop(columns=['Site']))
    
    batch_train_r = robjects.FactorVector(X_LR_temp['Site'])
    batch_test_r = robjects.FactorVector(X_HR_temp['Site'])
    
    # --- Fit CovBat on training data ---
    covbat_fit = fit_covbat(X_LR_r, batch_train_r)
    
    # --- Apply CovBat ---
    X_LR_harmonized_r = apply_covbat(covbat_fit, X_LR_r, batch_train_r)
    X_HR_harmonized_r = apply_covbat(covbat_fit, X_HR_r, batch_test_r)
    
    # --- Convert back to pandas ---
    with localconverter(robjects.default_converter + pandas2ri.converter):
        X_LR_harmonized = robjects.conversion.rpy2py(X_LR_harmonized_r)
        X_HR_harmonized = robjects.conversion.rpy2py(X_HR_harmonized_r)
    
    # --- Restore NaNs ---
    X_LR_harmonized[nan_indices_train] = np.nan
    X_HR_harmonized[nan_indices_test] = np.nan

    X_LR_harmonized_df = pd.DataFrame(X_LR_harmonized)
    column_names_LR = X_LR.columns.to_list()
    column_names_LR.remove("Site")
    X_LR_harmonized_df.columns = column_names_LR
    X_HR_harmonized_df = pd.DataFrame(X_HR_harmonized)
    column_names_HR = X_HR.columns.to_list()
    column_names_HR.remove("Site")
    X_HR_harmonized_df.columns = column_names_HR

    return X_LR_harmonized_df, X_HR_harmonized_df