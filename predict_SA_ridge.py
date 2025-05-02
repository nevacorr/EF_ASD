from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import time
from neurocombat_sklearn import CombatModel
from sklearn.model_selection import GridSearchCV

def tune_ridge_alpha(X, y, site_column='Site', sex_column='Sex'):
    combat = CombatModel()

    X_no_site_sex = X.drop(columns=[site_column, sex_column])

    # Fill NaNs with column means
    X_no_site_sex_filled = X_no_site_sex.fillna(X_no_site_sex.mean())

    sites = pd.Categorical(X[site_column]).codes
    sex = X[sex_column].values.reshape(-1, 1)

    # Harmonize
    X_combat = combat.fit_transform(X_no_site_sex_filled, sites.reshape(-1, 1))

    # Add sex back in
    X_final = np.hstack([sex, X_combat])

    # Define grid
    alphas = np.logspace(-4, 4, 20)
    ridge = Ridge()
    grid = GridSearchCV(ridge, {'alpha': alphas}, cv=10, scoring='r2')
    grid.fit(X_final, y)
    best_alpha = grid.best_params_['alpha']
    print(f"Best alpha from tuning: {best_alpha}")
    return best_alpha

def predict_SA_ridge(X, y, df, target, alpha_value, n_bootstraps):
    r2_test_all_bootstraps = []

    start_time = time.time()

    for b in range(n_bootstraps):
        print(f"Bootstrap iteration {b + 1}/{n_bootstraps}")

        train_predictions = np.zeros_like(y, dtype=np.float64)
        test_predictions = np.zeros_like(y, dtype=np.float64)
        train_counts = np.zeros_like(y, dtype=np.int64)

        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        for i, (train_index, test_index) in enumerate(kf.split(X, y)):

            combat = CombatModel()

            X_train = X.iloc[train_index].copy()
            y_train = y[train_index].copy()
            X_test = X.iloc[test_index].copy()
            y_test = y[test_index].copy()

            # Bootstrap the training data only
            bootstrap_indices = np.random.choice(len(train_index), size=len(train_index), replace=True)
            X_train_boot = X_train.iloc[bootstrap_indices].reset_index(drop=True)
            y_train_boot = y_train[bootstrap_indices]

            # Fill NaNs (important for Ridge)
            X_train_boot = X_train_boot.fillna(X_train_boot.mean())
            X_test = X_test.fillna(X_train_boot.mean())

            # Create Categorical object for the training set sites
            train_sites = pd.Categorical(X_train_boot['Site'])

            # Convert training sites to numeric codes (for harmonization)
            sites_train = train_sites.codes

            # Replace the 'Site' column in X_train with the codes
            X_train_boot['Site'] = sites_train

            # Apply the same categorical mapping to the test set sites
            test_sites = pd.Categorical(X_test['Site'], categories=train_sites.categories)
            sites_test = test_sites.codes

            # Replace the 'Site' column in X_test with the codes
            X_test['Site'] = sites_test

            # Keep a copy of Sex
            sex_train = X_train_boot['Sex'].values.reshape(-1,1)
            sex_test = X_test['Sex'].values.reshape(-1, 1)

            # Harmonize the training data
            X_train_boot_combat = combat.fit_transform(X_train_boot.drop(columns=['Site', 'Sex']), sites_train.reshape(-1, 1))

            # Harmonize the test data (using the same harmonization model fitted on the training data)
            X_test_combat = combat.transform(X_test.drop(columns=['Site', 'Sex']), sites_test.reshape(-1, 1))

            # Add sex values back into array for xgboost now that the brain measures have been harmonized
            X_train_boot_combat = np.hstack([sex_train,X_train_boot_combat])
            X_test_combat = np.hstack([sex_test, X_test_combat])

            # Initialize and fit Ridge regression
            ridge = Ridge(alpha=alpha_value)
            ridge.fit(X_train_boot_combat, y_train_boot)

            # Predict
            test_predictions[test_index] = ridge.predict(X_test_combat)
            train_predictions[train_index] += ridge.predict(X_train_boot_combat)

            train_counts[train_index] += 1

        # Correct training predictions
        train_predictions /= train_counts

        # Compute R²
        r2_test = r2_score(y, test_predictions)
        r2_train = r2_score(y, train_predictions)

        print(f"Ridge regression final performance. R2train = {r2_train:.3f}, R2test = {r2_test:.3f}")

        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60.0
        print(f"Bootstrap {b + 1}/{n_bootstraps} complete. Time to run this bootstrap: {elapsed_time:.2f} minutes")

        r2_test_all_bootstraps.append(r2_test)

    return r2_test_all_bootstraps