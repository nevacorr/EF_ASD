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

    mystop=1

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

def plot_top_shap_scatter_by_group(shap_feature_importance_df, all_shap_values, all_feature_values, all_group_labels, top_n=5):
    """
    Plots scatter plots for top N important features based on SHAP values,
    colored by group.

    Parameters:
    - shap_feature_importance_df: DataFrame with 'feature' and 'mean_abs_shap' columns
    - all_shap_values: numpy array of shape (n_samples, n_features)
    - all_feature_values: DataFrame of original feature values aligned with SHAP values
    - all_group_labels: Series or array of group labels aligned with rows
    - top_n: number of top features to plot (default=5)
    """

    all_feature_values = all_feature_values.drop(columns=['Site'])
    top_features = shap_feature_importance_df['feature'].head(top_n)

    for feature in top_features:
        # Get index of the feature
        feature_idx = list(shap_feature_importance_df['feature']).index(feature)

        # Extract x (feature values) and y (SHAP values)
        x_vals = all_feature_values[feature]
        y_vals = all_shap_values[:, feature_idx]

        x_vals = x_vals.reset_index(drop=True)
        all_group_labels = all_group_labels.reset_index(drop=True)

        # Plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=x_vals, y=y_vals, hue=all_group_labels, palette='Set2')
        plt.axhline(0, linestyle='--', color='gray', linewidth=1)
        plt.xlabel(f"{feature} value")
        plt.ylabel("SHAP value")
        plt.title(f"SHAP Scatter Plot: {feature}")
        plt.tight_layout()
        plt.show()


def plot_top_shap_distributions_by_group(shap_feature_importance_df, all_shap_values, all_group_labels, all_sex_labels, feature_names,
                                         top_n=5):
    """
    Plot violin plots of SHAP values for the top_n features, colored by group.

    Parameters:
    - shap_feature_importance_df: DataFrame with 'feature' and 'mean_abs_shap' columns
    - all_shap_values: numpy array (n_samples, n_features)
    - all_group_labels: pandas Series (n_samples,) with group labels
    - feature_names: list of feature names in order matching SHAP values
    - top_n: number of top features to plot
    """

    top_features = shap_feature_importance_df['feature'].head(top_n).tolist()

    for feature_name in top_features:
        feature_idx = feature_names.index(feature_name)
        shap_vals = all_shap_values[:, feature_idx]

        df = pd.DataFrame({
            "SHAP value": shap_vals,
            "Group": all_group_labels.reset_index(drop=True)  # <== reset index!
        })

        plt.figure(figsize=(8, 6))
        sns.violinplot(data=df, x="Group", y="SHAP value", inner="box", palette="Set2")
        plt.title(f"SHAP Distribution by Group for Feature: {feature_name}")
        plt.axhline(0, linestyle='--', color='gray')
        plt.tight_layout()
        plt.show()



def plot_shap_magnitude_histograms_equal_bins(all_shap_values, all_group_labels, all_sex_labels, feature_names,
                                              sex_feature_name='sex'):
    sex_idx = feature_names.index(sex_feature_name)
    sex_shap_vals = all_shap_values[:, sex_idx]

    sex_series = pd.Series(all_sex_labels).map({0: 'Female', 1: 'Male'}) if np.issubdtype(type(all_sex_labels[0]),
                                                                                          np.number) else pd.Series(
        all_sex_labels)
    group_series = pd.Series(all_group_labels)

    plot_df = pd.DataFrame({
        'Group': group_series,
        'Sex': sex_series,
        '|SHAP value|': np.abs(sex_shap_vals)
    })

    groups = plot_df['Group'].unique()
    sexes = ['Male', 'Female']
    colors = {'Male': 'blue', 'Female': 'green'}

    n_bins = 30
    fig, axes = plt.subplots(nrows=int(np.ceil(len(groups) / 3)), ncols=3,
                             figsize=(15, 5 * int(np.ceil(len(groups) / 3))))
    axes = axes.flatten()

    for i, grp in enumerate(groups):
        ax = axes[i]
        subset = plot_df[plot_df['Group'] == grp]

        # Determine common bins for this group's data
        data_min = subset['|SHAP value|'].min()
        data_max = subset['|SHAP value|'].max()
        bins = np.linspace(data_min, data_max, n_bins + 1)

        for sex in sexes:
            sex_data = subset[subset['Sex'] == sex]['|SHAP value|']
            ax.hist(sex_data, bins=bins, alpha=0.6, label=sex, color=colors[sex], density=True, histtype='bar',
                    edgecolor='black')

        ax.set_title(f'Group: {grp}')
        ax.set_xlabel('|SHAP value|')
        ax.set_ylabel('Count')
        ax.legend()

    # Remove unused subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Distribution of |SHAP| Values for Sex Feature by Group and Sex', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_shap_magnitude_by_sex_and_group(all_shap_values, all_group_labels, all_sex_labels, feature_names, sex_feature_name='sex'):
    """
    Plot boxplots of |SHAP| values for the sex feature, split by sex and diagnostic group.

    Parameters:
    - all_shap_values: np.ndarray of shape (n_samples, n_features)
    - all_group_labels: pandas Series or array-like of group labels (length = n_samples)
    - all_sex_labels: pandas Series or array-like of sex labels (e.g., 'Male'/'Female')
    - feature_names: list of feature names (length = n_features)
    - sex_feature_name: the exact name of the sex feature in feature_names
    """

    # Get index of the sex feature
    sex_feature_index = feature_names.index(sex_feature_name)

    # Get the SHAP values for just the sex feature
    sex_shap_values = all_shap_values[:, sex_feature_index]

    # Build DataFrame
    plot_df = pd.DataFrame({
        'Group': pd.Series(all_group_labels).reset_index(drop=True),
        'Sex': pd.Series(all_sex_labels).map({0: 'Female', 1: 'Male'}).reset_index(drop=True),
        '|SHAP value|': np.abs(sex_shap_values)
    })

    # Plot boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=plot_df, x='Group', y='|SHAP value|', hue='Sex', palette={'Male': 'blue', 'Female': 'green'})
    plt.title("SHAP Magnitude for Sex Feature by Group and Sex")
    plt.ylabel("|SHAP value| (magnitude)")
    plt.tight_layout()
    plt.show()

def plot_shap_magnitude_kde(all_shap_values, all_group_labels, all_sex_labels, feature_names, sex_feature_name='sex'):
    """
    Plot KDE plots of absolute SHAP values for the sex feature, separated by group and sex.

    Parameters:
    - all_shap_values: numpy array (n_samples, n_features)
    - all_group_labels: array-like of group labels (length n_samples)
    - all_sex_labels: array-like of sex labels coded as 0/1 or strings
    - feature_names: list of feature names
    - sex_feature_name: name of the sex feature in feature_names
    """

    sex_idx = feature_names.index(sex_feature_name)
    sex_shap_vals = all_shap_values[:, sex_idx]

    # Map numeric sex labels if needed
    sex_series = pd.Series(all_sex_labels).map({0: 'Female', 1: 'Male'}) if np.issubdtype(type(all_sex_labels[0]), np.number) else pd.Series(all_sex_labels)

    plot_df = pd.DataFrame({
        'Group': pd.Series(all_group_labels),
        'Sex': sex_series,
        '|SHAP value|': np.abs(sex_shap_vals)
    })

    g = sns.FacetGrid(plot_df, col="Group", hue="Sex", palette={'Male': 'blue', 'Female': 'green'}, col_wrap=3, height=4, sharex=True, sharey=True)
    g.map(sns.kdeplot, '|SHAP value|', fill=True, common_norm=False, alpha=0.6)
    g.add_legend()
    g.set_axis_labels("|SHAP value|", "Density")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('KDE of |SHAP| Values for Sex Feature by Group and Sex')
    plt.show()