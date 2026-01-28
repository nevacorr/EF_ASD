import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway


def cluster_hr_group(df, dx_group='HR', brain_cols=None, ef_cols=None,
                     method='kmeans', max_clusters=4, random_state=42):
    """
    Cluster HR kids based on subcortical brain volumes and examine EF profiles.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing HR_status, brain_cols, ef_cols.
    dx_group : str
        'HR+' or 'HR-' (group to subset for clustering)
    brain_cols : list
        List of brain volume column names
    ef_cols : list
        List of EF subscale column names
    method : str
        'kmeans' currently supported
    max_clusters : int
        Maximum number of clusters to try
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    df_group : pd.DataFrame
        Subsetted df with assigned 'cluster' column
    cluster_means : pd.DataFrame
        Mean brain volumes per cluster
    ef_means : pd.DataFrame
        Mean EF subscales per cluster
    """
    # 1️⃣ Subset HR group
    df_group = df[df['Risk'] == dx_group].copy()
    if df_group.empty:
        raise ValueError(f"No subjects found for HR group {dx_group}")

    # 2️⃣ Drop rows with NaNs in brain columns
    X = df_group[brain_cols].dropna()
    df_group = df_group.loc[X.index]  # align

    # 3️⃣ Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4️⃣ Find best number of clusters using silhouette
    best_score = -1
    best_k = 2
    best_labels = None

    for k in range(2, min(max_clusters, len(df_group)) + 1):
        if method == 'kmeans':
            model = KMeans(n_clusters=k, random_state=random_state)
            labels = model.fit_predict(X_scaled)
        else:
            raise ValueError("Only 'kmeans' supported currently")
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    print(f"Best number of clusters: {best_k}, silhouette score: {best_score:.3f}")
    df_group['cluster'] = best_labels

    # 5️⃣ Cluster means (brain volumes)
    cluster_means = df_group.groupby('cluster')[brain_cols].mean()
    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_means, annot=True, cmap='vlag')
    plt.title(f"{dx_group} Brain Volume Cluster Means")
    plt.show()

    # 6️⃣ EF comparison across clusters
    if ef_cols:
        ef_means = df_group.groupby('cluster')[ef_cols].mean()
        print("=== EF Subscale Means per Cluster ===")
        print(ef_means)

        print("\n=== ANOVA Across Clusters for EF Subscales ===")
        clusters = df_group['cluster'].unique()
        clusters.sort()
        for ef in ef_cols:
            samples = [df_group[df_group['cluster'] == c][ef].dropna() for c in clusters]
            F, p = f_oneway(*samples)
            print(f"{ef}: F={F:.2f}, p={p:.4f}")

        # Radar plot
        num_features = len(ef_cols)
        angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
        angles += angles[:1]

        plt.figure(figsize=(8, 8))
        for c in clusters:
            values = df_group[df_group['cluster'] == c][ef_cols].mean().tolist()
            values += values[:1]
            plt.polar(angles, values, label=f'Cluster {c}')
        plt.title(f"EF Profiles by Brain Cluster ({dx_group})")
        plt.legend(loc='upper right')
        plt.show()
    else:
        ef_means = None

    # 7️⃣ PCA scatter of brain volumes
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_group['cluster'], palette='tab10')
    plt.title(f"PCA of Brain Volumes ({dx_group})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

    return df_group, cluster_means, ef_means