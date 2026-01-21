import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# -----------------------------
# Helper function to prepare features
def compute_features(df, brain_cols, ef_cols, cluster_mode='absolute', time1=None, time2=None):
    if cluster_mode == 'absolute':
        features = []
        if brain_cols:
            features.append(df[[f"{col}_{time2}" for col in brain_cols]] if time2 else df[brain_cols])
        if ef_cols:
            features.append(df[[f"{col}_{time2}" for col in ef_cols]] if time2 else df[ef_cols])
        X = pd.concat(features, axis=1)
    elif cluster_mode == 'delta':
        if time1 is None or time2 is None:
            raise ValueError("For delta mode, time1 and time2 must be specified")
        brain_delta = df[[f"{col}_{time2}" for col in brain_cols]].values - df[
            [f"{col}_{time1}" for col in brain_cols]].values if brain_cols else np.empty((len(df), 0))
        ef_delta = df[[f"{col}_{time2}" for col in ef_cols]].values - df[
            [f"{col}_{time1}" for col in ef_cols]].values if ef_cols else np.empty((len(df), 0))
        X = pd.DataFrame(np.hstack([brain_delta, ef_delta]), columns=[f"Δ{col}" for col in brain_cols + ef_cols])
    else:
        raise ValueError("cluster_mode must be 'absolute' or 'delta'")
    return X


# -----------------------------
# Main clustering + summary workflow
def cluster_and_summarize(df, brain_cols=[], ef_cols=[], cluster_type='combined',
                          cluster_mode='absolute', time1=None, time2=None,
                          method='kmeans', max_clusters=6, hr_col='HR_status',
                          random_state=42):
    # Prepare features
    if cluster_type == 'brain':
        X = compute_features(df, brain_cols, [], cluster_mode, time1, time2)
    elif cluster_type == 'ef':
        X = compute_features(df, [], ef_cols, cluster_mode, time1, time2)
    elif cluster_type == 'combined':
        X = compute_features(df, brain_cols, ef_cols, cluster_mode, time1, time2)
    else:
        raise ValueError("cluster_type must be 'brain', 'ef', or 'combined'")

    feature_names = X.columns.tolist()

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine best number of clusters using silhouette score
    best_score = -1
    best_k = 2
    best_labels = None

    for k in range(2, min(max_clusters, len(df)) + 1):
        if method == 'kmeans':
            model = KMeans(n_clusters=k, random_state=random_state)
            labels = model.fit_predict(X_scaled)
        elif method == 'gmm':
            model = GaussianMixture(n_components=k, random_state=random_state)
            labels = model.fit_predict(X_scaled)
        else:
            raise ValueError("method must be 'kmeans' or 'gmm'")
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    print(f"Best number of clusters: {best_k}, silhouette score: {best_score:.3f}")
    df = df.copy()
    df['cluster'] = best_labels

    # -------------------------
    # Heatmap of cluster centroids
    cluster_means = df.groupby('cluster')[feature_names].mean()
    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_means, annot=True, cmap='vlag')
    plt.title(f"{cluster_type.capitalize()} ({cluster_mode}) Cluster Means")
    plt.show()

    # -------------------------
    # Radar plot
    num_features = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(8, 8))
    for c in cluster_means.index:
        values = cluster_means.loc[c].values.tolist()
        values += values[:1]
        plt.polar(angles, values, label=f'Cluster {c}')
    plt.title(f"{cluster_type.capitalize()} ({cluster_mode}) Cluster Profiles (Radar Plot)")
    plt.legend(loc='upper right')
    plt.show()

    # -------------------------
    # PCA scatter
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['cluster'], palette='tab10')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{cluster_type.capitalize()} ({cluster_mode}) PCA Scatter Plot")
    plt.legend(title='Cluster')
    plt.show()

    # -------------------------
    # HR composition per cluster
    print("=== HR Group Composition per Cluster ===")
    hr_summary = df.groupby('cluster')[hr_col].value_counts(normalize=True).unstack(fill_value=0)
    print(hr_summary)

    # -------------------------
    # EF subscale means per cluster
    if ef_cols:
        print("\n=== EF Subscale Means per Cluster ===")
        ef_means = df.groupby('cluster')[ef_cols].mean()
        print(ef_means)

        # ANOVA for each EF subscale
        print("\n=== ANOVA Across Clusters for EF Subscales ===")
        clusters = df['cluster'].unique()
        clusters.sort()
        for ef in ef_cols:
            samples = [df[df['cluster'] == c][ef].values for c in clusters]
            F, p = f_oneway(*samples)
            print(f"{ef}: F={F:.2f}, p={p:.4f}")

        # Radar plot for EF subscales
        num_features = len(ef_cols)
        angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
        angles += angles[:1]
        plt.figure(figsize=(8, 8))
        for c in clusters:
            values = df[df['cluster'] == c][ef_cols].mean().values.tolist()
            values += values[:1]
            plt.polar(angles, values, label=f'Cluster {c}')
        plt.title("EF Subscale Profiles by Cluster")
        plt.legend(loc='upper right')
        plt.show()
    else:
        ef_means = None

    return df, hr_summary, ef_means

def compute_lr_correlation_matrix(df):
    lr_pairs = {
        "Caudate": ("Caudate_Left_vol_VSA", "Caudate_Right_vol_VSA"),
        "Putamen": ("Putamen_Left_vol_VSA", "Putamen_Right_vol_VSA"),
        "Pallidum": ("Globus_Pallidus_Left_vol_VSA", "Globus_Pallidus_Right_vol_VSA"),
        "Thalamus": ("Thalamus_Left_vol_VSA", "Thalamus_Right_vol_VSA"),
        "Hippocampus": ("Hippocampus_Left_vol_VSA", "Hippocampus_Right_vol_VSA"),
        "Amygdala": ("Amygdala_Left_vol_VSA", "Amygdala_Right_vol_VSA"),

    }
    structures = list(lr_pairs.keys())
    corr_mat = pd.DataFrame(index=structures, columns=structures, dtype=float)

    for left_name, (l_col, _) in lr_pairs.items():
        for right_name, (_, r_col) in lr_pairs.items():
            valid = df[[l_col, r_col]].dropna()
            if len(valid) < 10:
                corr = np.nan
            else:
                corr = valid[l_col].corr(valid[r_col])
            corr_mat.loc[left_name, right_name] = corr

    return corr_mat

def plot_lr_heatmap(corr_mat, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_mat,
        vmin=0,
        vmax=1,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Pearson r"}
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()

