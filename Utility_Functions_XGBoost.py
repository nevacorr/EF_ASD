import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

def write_modeling_data_and_outcome_to_file(quick_run, metric, params, set_parameters_manually, target,
                                            df, r2_train, r2_test, best_params, elapsed_time):
    with open(f"{target}_{metric}_xgboost_run_results_summary.txt", "a") as f:
        # Write featueres and targets used
        f.write(f"####### Model performance summary ######\n")
        if quick_run == 1:
            f.write("This was a quick run to check code, not fit model\n")
        if set_parameters_manually == 0:
            f.write("Used hyperparameter optimization\n")
        elif set_parameters_manually ==1:
            f.write("Set parameter manually\n")
        f.write(f"Metric: {metric}\n")
        f.write(f"Target: {target}\n")
        feature_names = df.drop(columns=[target, 'test_predictions', 'train_predictions']).columns.tolist()
        f.write(f"Features: {', '.join(feature_names)}\n")
        f.write("Parameter specified\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Best Parameters: {best_params}\n")
        f.write("Performance metrics:\n")
        f.write(f"R2 train = {r2_train:.4f}\n")
        f.write(f"R2 test = {r2_test:.4f}\n")
        f.write(f"Run completion time: {elapsed_time:.2f}\n")

def plot_xgb_actual_vs_pred(metric, target, r2_train, r2_test, df, best_params):
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
            f"subsample={subsample:.2f} max_depth={max_depth}")

        # Show the plot
    plt.tight_layout()
    plt.savefig(f"{target}_{metric}_xgboost_actual_vs_predicted")
    plt.show(block=False)