import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def plot_correlations(df, title):
    correlation_matrix = df.corr()

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

def write_modeling_data_and_outcome_to_file(quick_run, metric, params, set_parameters_manually, target,
                                            X, r2_train, r2_test, best_params, bootstrap, elapsed_time):
    with open(f"{target}_{metric}_xgboost_run_results_summary.txt", "a") as f:
        now = datetime.now()
        f.write(f"Run Date and Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        # Write featueres and targets used
        f.write(f"####### Model performance summary ######\n")
        if quick_run == 1:
            f.write("This was a quick run to check code, not fit model\n")
        if set_parameters_manually == 0:
            f.write("Used hyperparameter optimization\n")
        elif set_parameters_manually ==1:
            f.write("Set parameter manually\n")
        if bootstrap ==1:
            f.write("Bootstrapped")
        f.write("Used harmonization by site\n")
        f.write(f"Metric: {metric}\n")
        f.write(f"Target: {target}\n")
        feature_names = X
        f.write(f"Features: {', '.join(feature_names)}\n")
        f.write("Parameter specified\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Best Parameters: {best_params}\n")
        f.write("Performance metrics:\n")
        f.write(f"R2 train = {r2_train:.4f}\n")
        f.write(f"R2 test = {r2_test:.4f}\n")
        f.write(f"Run completion time: {elapsed_time:.2f}\n")

def plot_xgb_actual_vs_pred(metric, target, r2_train, r2_test, df, best_params, show_plot):
    # Create subplots with 2 rows and 1 column
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Define a list of prediction types and corresponding R2 values
    prediction_data = [
        ("test", r2_test, df["test_predictions"]),
        ("train", r2_train, df["train_predictions"])
    ]
    colsample_bytree = best_params['colsample_bytree']
    n_estimators = best_params['n_estimators']
    min_child_weight = best_params['min_child_weight']
    gamma = best_params['gamma']
    eta = best_params['eta']
    subsample = best_params['subsample']
    max_depth = best_params['max_depth']

    for i, (data_type, r2, predictions) in enumerate(prediction_data):
        sns.scatterplot(x=df[target], y=predictions, ax=axes[i])
        axes[i].plot([min(df[target]), max(df[target])], [min(df[target]), max(df[target])], linestyle="--",
                     color="red")
        axes[i].set_xlabel(f"Actual {metric} {target}")
        axes[i].set_ylabel(f"Predicted {metric} {target}")
        axes[i].set_title(
            f"{data_type.capitalize()} {metric} Predictions\nR2: {r2:.2f}\n colsample_by_tree={colsample_bytree:.2f} "
            f"n_estimators={n_estimators} min_child_weight={min_child_weight} gamma={gamma:.2f}\n eta={eta:.2e} "
            f"subsample={subsample:.2f} max_depth={max_depth} harmonized by Site")

        # Show the plot
    plt.tight_layout()
    plt.savefig(f"{target}_{metric}_xgboost_actual_vs_predicted")
    if show_plot==1:
        plt.show(block=False)


def generate_bootstrap_indices(n_rows, n_iterations=100):

    np.random.seed(42)

    indices_list = []
    for _ in range(n_iterations):
        bootstrap_indices = np.random.choice(n_rows, size=n_rows, replace=True)
        indices_list.append(bootstrap_indices)

    return indices_list

def calculate_percentile(r2_test_array, alpha):

    lower_bound = np.percentile(r2_test_array, alpha * 100)

    if lower_bound > 0:
        result_text = f"R² is significantly greater than 0 at the {int(alpha * 100)}% level (one-tailed)"
    else:
        result_text = f"R² is NOT significant at the 5% level (one-tailed)"

    print(f"{(1 - alpha) * 100}% lower confidence bound: {lower_bound:.4f}")

    # Get value below which alpha*100 % of the data fall
    percentile_value = np.percentile(r2_test_array, alpha * 100)

    return result_text, percentile_value

def plot_r2_distribution(r2_test_array, result_text, percentile_value,
                         target, metric, alpha, n_bootstraps, alg):
    # Plot the distribution
    plt.figure(figsize=(10, 8))
    sns.histplot(r2_test_array, bins=30, kde=True, color='skyblue')

    # Add vertical line at the 5th percentile
    plt.axvline(percentile_value, color='red', linestyle='--', linewidth=2,
                label=f'5th percentile = {percentile_value:.3f}')

    # Add title and labels
    plt.title(
        f'{target} predicted from {metric} with {alg}\nBootstrap r²test Distribution '
        f'with {alpha * 100:.0f}% Percentile Marked\n{result_text}\nnbootstraps={n_bootstraps}')
    plt.xlabel('r² test')
    plt.ylabel('Frequency')

    # Add legend
    plt.legend()

    # Show plot
    plt.show()

def aggregate_feature_importances(feature_importance_list, feature_names, n_boot, outputfilename, top_n=10, plot=True):
    """
    Aggregates and plots feature importances across bootstraps.

    Parameters:
        feature_importance_list (list): List of arrays from model.feature_importances_ across bootstraps.
        feature_names (list or pd.Index): Names of features corresponding to the importances.
        top_n (int): Number of top features to display.
        plot (bool): Whether to generate a bar plot.

    Returns:
        pd.DataFrame: Sorted dataframe of mean and std of feature importances.
    """
    # Convert to numpy array: shape (n_bootstraps, n_features)
    importance_array = np.array(feature_importance_list)
    mean_importances = importance_array.mean(axis=0)
    std_importances = importance_array.std(axis=0)

    # Create a DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_importance': mean_importances,
        'std_importance': std_importances
    })

    # Sort by mean importance
    sorted_df = importance_df.sort_values(by='mean_importance', ascending=False).reset_index(drop=True)

    sorted_df.to_csv(outputfilename)

    if plot:
        top_features = sorted_df.head(top_n)
        plt.figure(figsize=(10, 6))
        plt.barh(top_features['feature'], top_features['mean_importance'], xerr=top_features['std_importance'], color='skyblue')
        plt.gca().invert_yaxis()
        plt.xlabel('Mean Importance')
        plt.title(f'Top {top_n} Feature Importances (Bootstrapped XGBoost {n_boot} bootstraps)')
        plt.tight_layout()
        plt.show()

    return sorted_df