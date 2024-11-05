import pandas as pd
import matplotlib.pyplot as plt
import math

def plot_data_histograms(working_dir, df):
    # Define the number of columns for each page of subplots
    plots_per_page = 9
    num_columns = 3
    num_rows = math.ceil(plots_per_page / num_columns)

    # Loop through the DataFrame columns in chunks
    for i in range(0, len(df.columns), plots_per_page):
        # Set up the figure with the desired grid size
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 4 * num_rows))
        axes = axes.flatten()  # Flatten to easily index each subplot

        # Plot each column in the current chunk
        for j, column in enumerate(df.columns[i:i + plots_per_page]):
            data = df[column]
            non_nan_data = data.dropna()  # Drop NaNs for plotting
            nan_count = data.isna().sum()  # Count NaN values

            # Check if the column is numeric
            if pd.api.types.is_numeric_dtype(non_nan_data):
                axes[j].hist(non_nan_data, bins=10, color='blue', edgecolor='black')
                axes[j].set_title(f'{column}', fontsize=10)
                axes[j].set_xlabel(column)
                axes[j].set_ylabel('Count')
                # Add a text box with the count of NaN values
                axes[j].text(0.95, 0.95, f'NaN count: {nan_count}',
                             transform=axes[j].transAxes, ha='right', va='top',
                             bbox=dict(facecolor='white', alpha=0.5))

            # Check if the column is categorical or has few unique values
            elif pd.api.types.is_object_dtype(non_nan_data) or non_nan_data.nunique() <= 10:
                counts = non_nan_data.value_counts()
                # Include NaN count in the bar plot
                counts['NaN'] = nan_count
                counts.plot(kind='bar', ax=axes[j], color='blue', edgecolor='black')
                axes[j].set_title(f'{column}', fontsize=10)
                axes[j].set_xlabel(column)
                axes[j].set_ylabel('Count')

        # Hide any unused subplots in the grid
        for k in range(j + 1, len(axes)):
            fig.delaxes(axes[k])

        # Adjust layout and show the figure for the current page
        plt.tight_layout()
        plt.savefig(f'{working_dir}/IBIS_Raw_Data_Histograms_Fig{int(i/plots_per_page)}.png')
        plt.show(block=False)
