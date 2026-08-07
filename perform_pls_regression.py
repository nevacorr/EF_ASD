import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import r2_score


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

def perform_pls_regression(final_brain_df, brain_cols, df_hr_z, ef_col, perform_norm_modeling,
                           max_components=3, n_permutations=1000):

    df_ef = final_brain_df[['Identifiers', ef_col]].copy()
    df_all = pd.merge(df_ef, df_hr_z, on="Identifiers", how="inner")
    df_all = df_all.dropna().reset_index(drop=True)

    if perform_norm_modeling:
        brain_cols_z = [col + '_z' for col in brain_cols]
    else:
        brain_cols_z = brain_cols

    X = df_all[brain_cols_z].copy()
    y = df_all[ef_col].copy()

    # ── Train/Test split ──────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ── Standardize (fit on train only) ───────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── 1. Select optimal components via CV ───────────────────────────────────
    optimal_components, min_mse, mse_values, component_range = optimise_pls_cv(
        X_train_scaled, y_train, max_components
    )
    print(f"Optimal components: {optimal_components}, CV MSE: {min_mse:.4f}")

    # ── Fit final model ───────────────────────────────────────────────────────
    pls_model = PLSRegression(n_components=optimal_components)
    pls_model.fit(X_train_scaled, y_train)
    y_pred = pls_model.predict(X_test_scaled).ravel()

    r_squared = pls_model.score(X_test_scaled, y_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"R-Squared: {r_squared:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

    # ── 2. Permutation test ───────────────────────────────────────────────────
    perm_r2 = []
    for _ in tqdm(range(n_permutations), desc="Permutation test"):
        y_perm = np.random.permutation(y_test)
        perm_r2.append(r2_score(y_perm, y_pred))

    perm_r2 = np.array(perm_r2)
    p_value = (np.sum(perm_r2 >= r_squared) + 1) / (n_permutations + 1)
    print(f"Permutation p-value: {p_value:.3f}")

    # ── 3. Feature importance ─────────────────────────────────────────────────
    weights = pd.Series(pls_model.x_weights_[:, 0], index=brain_cols_z)
    top_features = weights.abs().sort_values(ascending=False).head(10)
    print("\nTop 10 features (by absolute weight on component 1):")
    print(weights.loc[top_features.index].round(6).to_string())

    # ── Predicted vs Actual plot ──────────────────────────────────────────────
    plt.figure()
    plt.scatter(y_test, y_pred, c='blue', label='Actual vs Predicted')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', c='red', label='Perfect Prediction')
    plt.xlabel(f"Actual {ef_col}")
    plt.ylabel(f"Predicted {ef_col}")
    plt.title(
        f"PLS Regression: Predicted vs Actual {ef_col}\n"
        f"R²={r_squared:.3f}, p={p_value:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ── Feature importance bar plot ───────────────────────────────────────────
    top_n = 10
    top_weights = weights.abs().sort_values(ascending=False).head(top_n)
    top_weights_signed = weights.loc[top_weights.index]

    colors = ['steelblue' if w > 0 else 'tomato' for w in top_weights_signed]

    plt.figure(figsize=(8, 5))
    plt.barh(top_weights_signed.index[::-1], top_weights_signed.values[::-1], color=colors[::-1])
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xlabel("PLS Weight (Component 1)")
    plt.title(f"Top {top_n} Brain Features — {ef_col}")
    plt.tight_layout()
    plt.show()

    return pls_model, r_squared, p_value, weights

    mystop=1