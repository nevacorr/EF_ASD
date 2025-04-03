
import os
import pandas as pd

def load_subcortical_data(filepath):
    # Get all the CSV filenames in the directory
    csv_files = [f for f in os.listdir(filepath) if f.endswith('.csv')]

    # Create an empty dictionary to store dataframes
    dfs = {}

    # Load each CSV file into a separate dataframe
    for file in csv_files:
        file_name = os.path.join(filepath, file)
        # Read each CSV into a DataFrame and store it in the dictionary
        dfs[file] = pd.read_csv(file_name)

    mystop=1

    # Example: access a specific DataFrame
    # example_file = "IBIS_v3.13_Amygdala_2020May5_V12V24only.csv"
    # df = dfs.get(example_file)