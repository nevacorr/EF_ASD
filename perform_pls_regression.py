import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import StandardScaler


def optimise_pls_cv(X, y, max_components):
    mse_values = []
    component_range = range(1, max_components + 1)

    # Use k-fold cross-validation for more reliable results than leave-one-out
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    for n_components in component_range:
        pls = PLSRegression(n_components=n_components)

        # Calculate predicted y values using cross-validation
        y_cv = cross_val_predict(pls, X, y, cv=cv)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y, y_cv)
        mse_values.append(mse)

    # Find the optimal number of components that minimizes MSE
    optimal_components = component_range[np.argmin(mse_values)]
    min_mse = np.min(mse_values)

    plt.plot(component_range, mse_values, marker='o')
    plt.xlabel("Number of PLS Components")
    plt.ylabel("CV MSE")
    plt.title("PLS Component Selection")
    plt.show()

    return optimal_components, min_mse, mse_values, component_range


def perform_pls_regression(final_brain_df, brain_cols, df_hr_z, ef_col):

    df_ef=final_brain_df[['Identifiers', ef_col]].copy()
    df_all=pd.merge(df_ef, df_hr_z,  on="Identifiers", how="inner")
    df_all=df_all.dropna().reset_index(drop=True)

    # Define features and target variable
    brain_cols_z = [col + '_z' for col in brain_cols]
    X = df_all[brain_cols_z].copy()
    y = df_all[ef_col].copy()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform scaling of the features
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    max_components = 10
    optimal_components, min_mse, mse_values, component_range = optimise_pls_cv(X_train_scaled, y_train, max_components)

    # Initialize PLS model with desired number of components
    pls_model = PLSRegression(n_components=optimal_components)

    # Fit the  model on the train set
    pls_model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = pls_model.predict(X_test_scaled).ravel()

    # Evaluate the model performance
    r_squared = pls_model.score(X_test_scaled, y_test)
    print(f"R-Squared Error: {r_squared}")
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Visualize predicted vs actual values with different colors
    plt.scatter(y_test, y_pred, c='blue', label='Actual vs Predicted')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', c='red', label='Perfect Prediction')
    plt.xlabel(f"Actual {ef_col}")
    plt.ylabel(f"Predicted {ef_col}")
    plt.title(f"PLS Regression: Predicted vs Actual {ef_col}")
    plt.legend()
    plt.show()

    mystop=1