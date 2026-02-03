import matplotlib.pyplot as plt
import seaborn as sns
import math


def plot_feature_histograms(df, feature_cols, max_cols=5):
    """
    Plots histograms with KDE for any number of features in df.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features.
    feature_cols : list of str
        List of column names to plot.
    max_cols : int
        Maximum number of columns in the subplot grid.
    """
    n_features = len(feature_cols)

    # Compute number of rows and columns
    n_cols = min(max_cols, n_features)
    n_rows = math.ceil(n_features / n_cols)

    # Scale figure size automatically
    plt.figure(figsize=(4 * n_cols, 3 * n_rows))

    # Adjust font size depending on number of features
    title_fs = max(10, 20 - n_features // 5)
    label_fs = max(8, 16 - n_features // 5)

    for i, feature in enumerate(feature_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df[feature], bins=20, kde=True, color='skyblue')
        plt.title(feature, fontsize=title_fs)
        plt.xlabel("Z-score", fontsize=label_fs)
        plt.ylabel("Count", fontsize=label_fs)

    plt.tight_layout()
    plt.show()
