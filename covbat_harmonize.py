import os
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from Utility_Functions_XGBoost import impute_by_site_median_with_nan_indices
import rpy2.robjects as ro
os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
robjects.r('source("~/R_Projects/IBIS_EF_xgboost/covbat_wrapper_for_use_with_python.R")')

def covbat_harmonize(X_train, X_test):
    
    # Assign covbat functions
    fit_covbat = robjects.r['fit_covbat']
    apply_covbat = robjects.r['apply_covbat']
    
    #  Replace NaN values with column median for harmonization
    fcols = X_train.columns.difference(['Site', 'Sex'])
    
    (X_train_temp, X_test_temp, nan_indices_train, nan_indices_test) = impute_by_site_median_with_nan_indices(
        X_train,
        X_test,
        feature_cols=fcols,
        site_col='Site'
    )
    
    # Keep a copy of Sex
    sex_train = X_train_temp['Sex'].values.reshape(-1, 1)
    sex_test = X_test_temp['Sex'].values.reshape(-1, 1)
    
    # --- Convert to R data frames ---
    with localconverter(robjects.default_converter + pandas2ri.converter):
        X_train_r = robjects.conversion.py2rpy(X_train_temp.drop(columns=['Site', 'Sex']))
        X_test_r = robjects.conversion.py2rpy(X_test_temp.drop(columns=['Site', 'Sex']))
    
    batch_train_r = robjects.FactorVector(X_train_temp['Site'])
    batch_test_r = robjects.FactorVector(X_test_temp['Site'])
    
    # --- Fit CovBat on training data ---
    covbat_fit = fit_covbat(X_train_r, batch_train_r)
    
    # --- Apply CovBat ---
    X_train_harmonized_r = apply_covbat(covbat_fit, X_train_r, batch_train_r)
    X_test_harmonized_r = apply_covbat(covbat_fit, X_test_r, batch_test_r)
    
    # --- Convert back to pandas ---
    with localconverter(robjects.default_converter + pandas2ri.converter):
        X_train_harmonized = robjects.conversion.rpy2py(X_train_harmonized_r)
        X_test_harmonized = robjects.conversion.rpy2py(X_test_harmonized_r)
    
    # --- Restore NaNs ---
    X_train_harmonized[nan_indices_train] = np.nan
    X_test_harmonized[nan_indices_test] = np.nan
    
    # Add sex values back into array for xgboost now that the brain measures have been harmonized
    X_train_harmonized = np.hstack([sex_train, X_train_harmonized])
    X_test_harmonized = np.hstack([sex_test, X_test_harmonized])

    return X_train_harmonized, X_test_harmonized