import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_hyperparameter_repeats(best_params_list, r2_vals, top_n=5):
    """
    Analyze repeated hyperparameter tuning results.

    Args:
        best_params_list: list of dicts, best hyperparameters from each repeat
        r2_vals: array-like, validation R² for each repeat
        top_n: int, number of top repeats to print

    Returns:
        best_params: dict, hyperparameter set from repeat with highest R²
    """
    # Convert list of dicts into DataFrame
    params_df = pd.DataFrame(best_params_list)
    params_df['R2_val'] = r2_vals

    # Sort by validation R² descending
    params_df_sorted = params_df.sort_values('R2_val', ascending=False)
    print(f"Top {top_n} repeats by validation R²:")
    print(params_df_sorted.head(top_n))

    # Plot validation R² per repeat
    plt.figure(figsize=(10,5))
    sns.barplot(x=range(len(r2_vals)), y=r2_vals)
    plt.xlabel("Repeat")
    plt.ylabel("Validation R²")
    plt.title("Validation R² for each repeat")
    plt.show()

    # Heatmap of numeric hyperparameters vs R²
    numeric_cols = params_df.select_dtypes(include='number').columns.tolist()
    plt.figure(figsize=(10,6))
    sns.heatmap(params_df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation between numeric hyperparameters and validation R²")
    plt.show()

    # Return the best hyperparameters (highest validation R²)
    best_idx = params_df['R2_val'].idxmax()
    best_params = best_params_list[best_idx]

    print("\nBest hyperparameter combination (highest R²):")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    return best_params