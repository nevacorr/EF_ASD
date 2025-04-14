import pandas as pd
import re
import matplotlib.pyplot as plt


# Function to extract metrics and group feature presence from the text
def extract_performance_and_group(text):
    # Regular expressions to capture key parts
    pattern_features = r"Features: (.*?)\n"
    pattern_metrics = r"R2 train = (\d+\.\d+)\s*R2 test = (\d+\.\d+)"

    # Extract features and metrics
    features = re.findall(pattern_features, text)
    metrics = re.findall(pattern_metrics, text)

    # Prepare a list to store the data
    data = []

    for i in range(len(features)):
        feature_list = features[i].split(", ")
        r2_train, r2_test = metrics[i]

        # Check if the Group feature is included
        has_group = any(group in feature_list for group in ['Group_HR+', 'Group_HR-', 'Group_LR-'])

        # Add the results to the data list
        data.append({
            'R2_train': float(r2_train),
            'R2_test': float(r2_test),
            'has_group': has_group
        })

    return pd.DataFrame(data)

# Read the content of the text file
file_path = '/Users/nevao/PycharmProjects/EF_ASD/BRIEF2_GEC_raw_score_subcort_xgboost_run_results_summary.txt'
title_str = 'BRIEF2_GEC_raw_score_subcort'
with open(file_path, 'r') as file:
    text = file.read()

# Extract data and put it into a DataFrame
df = extract_performance_and_group(text)

# Separate the data by whether the "Group" feature was included
df_with_group = df[df['has_group']]
df_without_group = df[~df['has_group']]


# Plot the results
def plot_performance_scatter(df_with_group, df_without_group, title_str):
    plt.figure(figsize=(10, 6))

    # Initialize plot lines for the legend
    line_with_group, = plt.plot([], [], color='blue', marker='o', markersize=6, label='With Group')
    line_without_group, = plt.plot([], [], color='green', marker='o', markersize=6, label='Without Group')

    # Scatter plot for R2_train and R2_test for models with Group
    for i in df_with_group.index:
        plt.plot([i, i], [df_with_group['R2_train'][i], df_with_group['R2_test'][i]], color='blue', marker='o', markersize=6)
        plt.text(i, df_with_group['R2_train'][i], f'R2_train={df_with_group["R2_train"][i]:.2f}', ha='center', va='bottom', fontsize=9, color='blue')
        plt.text(i, df_with_group['R2_test'][i], f'R2_test={df_with_group["R2_test"][i]:.2f}', ha='center', va='top', fontsize=9, color='blue')

    # Scatter plot for R2_train and R2_test for models without Group
    for i in df_without_group.index:
        plt.plot([i, i], [df_without_group['R2_train'][i], df_without_group['R2_test'][i]], color='green', marker='o', markersize=6)
        plt.text(i, df_without_group['R2_train'][i], f'R2_train={df_without_group["R2_train"][i]:.2f}', ha='center', va='bottom', fontsize=9, color='green')
        plt.text(i, df_without_group['R2_test'][i], f'R2_test={df_without_group["R2_test"][i]:.2f}', ha='center', va='top', fontsize=9, color='green')

    # Add labels and title
    plt.xlabel('Model Number')
    plt.ylabel('R2 Score')
    plt.title(f'{title_str}\nR2 Train vs R2 Test: With vs. Without Group Feature')

    # Add a legend to explain the colors, positioned outside the plot on the right
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10, borderpad=1, borderaxespad=1)

    # Show the plot
    plt.tight_layout()  # Ensures the plot doesn't get clipped
    plt.show()

# Call the function to plot
plot_performance_scatter(df_with_group, df_without_group, title_str)

mystop=1

