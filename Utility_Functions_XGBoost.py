import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_data(filepath, filename, target):

    df = pd.read_csv(f"{filepath}/{filename}")

    columns_to_exclude = ["CandID", "Identifiers", "Combined_ASD_DX", "Risk", "Group", "AB_12_Percent", "AB_24_Percent",
                          "BRIEF2_GEC_raw_score", "BRIEF2_GEC_T_score", "DCCS_Standard_Age_Corrected", "ICV_V12",
                          "ICV_V24", "totTiss_V12", "totTiss_V24"]

    df.drop(columns=columns_to_exclude, inplace=True)

    # Keep only rows where the response variable is not NA
    df = df[df[target].notna()]

    # Encode Sex column Female = 0 Male = 1
    df["Sex"] = df["Sex"].replace({"Female": 0, "Male": 1})

    return df

def plot_correlations(df, target, title):
    df_features = df.drop(columns=[target])
    correlation_matrix = df_features.corr()

    # Plot heatmap of correlation coefficients
    plt.figure(figsize=(11, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=1, yticklabels=1)
    # Adjust the margins
    plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.3)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def remove_collinearity(df, threshold):
    # Calculate the correlation matrix
    correlation_matrix = df.corr().abs()

    # Get the upper triangle of the correlation matrix
    upper_triangle = correlation_matrix.where(
        pd.np.triu(pd.np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    # Identify columns to drop
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    # Drop the correlated features
    df_reduced = df.drop(columns=to_drop)

    return df_reduced


