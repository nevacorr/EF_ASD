import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from helper_functions_clustering import plot_brain_vs_age_by_sex_from_model

def normative_clustering(df, group_col='Group', lr_label='LR-', hr_labels=['HR+', 'HR-'],
                         brain_cols=None, ef_cols=None, covariates=['Sex', 'Age'],
                         n_clusters=2, random_state=42):
    """
    1. Fit normative models on LR kids
    2. Compute z-scores for HR kids
    3. Cluster HR kids based on z-scores
    4. Summarize clusters and optionally EF
    """

    # --------------- 1. Split dataframe ---------------
    df_lr = df[df[group_col] == lr_label].copy()
    df_hr = df[df[group_col].isin(hr_labels)].copy()

    # --------------- 2. Compute z-scores ---------------
    z_cols = []
    df_hr_z = pd.DataFrame()
    df_hr_z['CandID'] = df_hr['CandID']
    df_hr_z.reset_index(inplace=True, drop=True)
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

    # --------------- 3. Cluster HR kids based on z-scores ---------------
    X = df_hr_z[z_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df_hr['cluster'] = kmeans.fit_predict(X_scaled)

    # Optional: silhouette score
    score = silhouette_score(X_scaled, df_hr['cluster'])
    print(f"Silhouette score: {score:.3f}")

    # --------------- 4. Summarize cluster profiles ---------------
    print("\n=== Cluster Mean Z-scores (Brain Measures) ===")
    cluster_means = df_hr.groupby('cluster')[z_cols].mean()
    print(cluster_means.round(2))

    # Heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(cluster_means, annot=True, cmap='vlag')
    plt.title("HR Clusters: Mean Z-scores of Brain Measures")
    plt.show()

    # EF summary
    if ef_cols:
        print("\n=== EF Means per Cluster ===")
        ef_means = df_hr.groupby('cluster')[ef_cols].mean()
        print(ef_means.round(2))

        # Radar plot for EF
        num_features = len(ef_cols)
        angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
        angles += angles[:1]

        plt.figure(figsize=(8,8))
        for c in cluster_means.index:
            values = df_hr[df_hr['cluster'] == c][ef_cols].mean().values.tolist()
            values += values[:1]
            plt.polar(angles, values, label=f'Cluster {c}')
        plt.title("EF Profiles by HR Cluster")
        plt.legend(loc='upper right')
        plt.show()
    else:
        ef_means = None

    return df_hr, cluster_means, ef_means
