import re
import pandas as pd


# Function to parse the OrderedDict
def parse_ordered_dict(param_str):
    """Parse the OrderedDict string to a dictionary."""
    param_pairs = re.findall(r"\('(\w+)', ([^()]+?)\)", param_str)
    return {k: eval(v) for k, v in param_pairs}


# Read in the file
with open('/Users/nevao/PycharmProjects/EF_ASD/BRIEF2_GEC_raw_score_subcort_xgboost_run_results_summary.txt',
          'r') as file:
    text = file.read()

# Split into model blocks and remove empty entries
model_blocks = text.split('####### Model performance summary ######')
model_blocks = [block.strip() for block in model_blocks if block.strip()]

# Extract info from each block
data = []
for block in model_blocks:
    row = {}

    # Check if Group variables are included in features
    row['GroupIncluded'] = 'Group_HR+' in block

    # Extract performance metrics
    r2_train = re.search(r'R2 train = ([\d.]+)', block)
    r2_test = re.search(r'R2 test = ([\d.]+)', block)
    runtime = re.search(r'Run completion time: ([\d.]+)', block)

    row['R2_train'] = float(r2_train.group(1)) if r2_train else None
    row['R2_test'] = float(r2_test.group(1)) if r2_test else None
    row['RunTime'] = float(runtime.group(1)) if runtime else None

    # Extract parameter ranges from the text
    param_ranges_match = re.search(r'Parameter specified\n(.*?)\nBest Parameters', block, re.DOTALL)
    param_ranges = {}  # Initialize the param_ranges dictionary
    if param_ranges_match:
        param_ranges_str = param_ranges_match.group(1)
        for line in param_ranges_str.split('\n'):
            match = re.match(r'(\w+):\s*\((.*?)\)', line.strip())
            if match:
                param_name, param_range = match.groups()
                # Try converting ranges to actual tuples or intervals
                try:
                    param_ranges[param_name] = eval(param_range)  # Convert (min, max) to a tuple
                except:
                    param_ranges[param_name] = param_range
        print(f"Extracted param_ranges: {param_ranges}")

    # Add param_ranges to the row dictionary
    row.update(param_ranges)

    # Extract best parameters using OrderedDict format
    param_match = re.search(r'Best Parameters:\s*OrderedDict\(\[(.*?)\]\)', block, re.DOTALL)
    if param_match:
        param_str = param_match.group(1)
        param_dict = parse_ordered_dict(param_str)
        # Append '_best' to the parameter name for best parameters
        for param_name, best_value in param_dict.items():
            row[f'{param_name}_best'] = best_value

    # Append the row to the data list
    data.append(row)

# Create a DataFrame
df = pd.DataFrame(data)

# Show the full DataFrame
pd.set_option('display.max_columns', None)
print(df)

df.to_csv("BRIEF2_GEC_raw_score_subcort_xgboost_results_dataframe.csv")