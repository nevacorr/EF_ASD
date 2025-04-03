import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_clean_data(filepath, filename, target, include_group):

    df = pd.read_csv(f"{filepath}/{filename}")

    if include_group:
        columns_to_exclude = ["CandID", "Identifiers", "Combined_ASD_DX", "Risk", "AB_12_Percent", "AB_24_Percent",
                              "BRIEF2_GEC_raw_score", "BRIEF2_GEC_T_score", "DCCS_Standard_Age_Corrected", "ICV_V12",
                              "ICV_V24", "totTiss_V12", "totTiss_V24"]
    else:
        columns_to_exclude = ["CandID", "Group", "Identifiers", "Combined_ASD_DX", "Risk", "AB_12_Percent", "AB_24_Percent",
                              "BRIEF2_GEC_raw_score", "BRIEF2_GEC_T_score", "DCCS_Standard_Age_Corrected", "ICV_V12",
                              "ICV_V24", "totTiss_V12", "totTiss_V24"]

    df.drop(columns=columns_to_exclude, inplace=True)

    # Keep only rows where the response variable is not NA
    df = df[df[target].notna()]

    # Encode Sex column Female = 0 Male = 1
    df["Sex"] = df["Sex"].replace({"Female": 0, "Male": 1})

    if include_group:
        # One-hot encode the 'Group' column
        df = pd.get_dummies(df, columns=['Group'], drop_first=False)

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

def load_and_clean_dti_data(dir, datafilename, vol_dir, voldatafile, target, include_group):

    dti_df = pd.read_csv(f"{dir}/{datafilename}")

    vol_df = pd.read_csv(f"{vol_dir}/{voldatafile}")

    vol_df = vol_df[vol_df['CandID'].notna()]

    vol_df['CandID'] = vol_df['CandID'].astype('int64')

    vol_df = vol_df.loc[:, ~vol_df.columns.str.contains('GM|WM|ICV|totTiss', regex=True)]

    vol_df.drop(columns=['Identifiers'], inplace=True)

    dti_df.drop(columns=['Visit_label', 'CandID_Visit', 'dMRI_protocol', 'FileID'], inplace=True)

    dti_df = dti_df.loc[:, ~dti_df.columns.str.contains('Optic|Motor|Fornix|CorticoSpinal|UNC|Reticular|ILF|CT|CF|CC|Temporo|IFOF', regex=True)]

    dti_df.replace({'.': np.nan, '': np.nan}, inplace=True)

    dti_df[dti_df.columns.difference(['CandID'])] = (
        dti_df[dti_df.columns.difference(['CandID'])].apply(pd.to_numeric, errors='coerce'))

    merged_df = vol_df.merge(dti_df, on="CandID", how="left")

    merged_df.drop(columns=['CandID'], inplace=True)

    # Keep only rows where the response variable is not NA
    merged_df = merged_df[merged_df[target].notna()]

    # Encode Sex column Female = 0 Male = 1
    merged_df["Sex"] = merged_df["Sex"].replace({"Female": 0, "Male": 1})

    if include_group:
        columns_to_exclude = ["Combined_ASD_DX", "Risk", "AB_12_Percent", "AB_24_Percent",
                              "BRIEF2_GEC_raw_score", "BRIEF2_GEC_T_score", "DCCS_Standard_Age_Corrected"]
    else:
        columns_to_exclude = ["Combined_ASD_DX","Group", "Risk", "AB_12_Percent", "AB_24_Percent",
                              "BRIEF2_GEC_raw_score", "BRIEF2_GEC_T_score", "DCCS_Standard_Age_Corrected"]

    merged_df.drop(columns=columns_to_exclude, inplace=True)

    if include_group:
        # One-hot encode the 'Group' column
        merged_df = pd.get_dummies(merged_df, columns=['Group'], drop_first=False)

    merged_df = merged_df.dropna(subset=[col for col in merged_df.columns if col not in ['Sex', target]], how='all')

    return merged_df

def write_modeling_data_and_outcome_to_file(quick_run, metric, params, set_parameters_manually, target,
                                            df, r2_train, r2_test, best_params, elapsed_time):
    with open(f"{target}_{metric}_features_and_target.txt", "a") as f:
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