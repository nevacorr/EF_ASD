import re
import pandas as pd
def parse_ordered_dict(text):
    """Safely parse the OrderedDict string into a Python dict"""
    if not text:
        return {}
    try:
        pairs = re.findall(r"\('([^']+)'\s*,\s*([^\)]+)\)", text)
        parsed = {}
        for k, v in pairs:
            try:
                parsed[k] = eval(v)
            except:
                parsed[k] = v.strip()
        return parsed
    except Exception as e:
        print(f"Error parsing OrderedDict: {e}")
        return {}

# Step 1: Read in the file
with open('/Users/nevao/PycharmProjects/EF_ASD/BRIEF2_GEC_raw_score_subcort_xgboost_run_results_summary.txt', 'r') as file:
    text = file.read()

# Step 2: Split into individual model results
model_blocks = text.split("####### Model performance summary ######")
model_blocks = [block.strip() for block in model_blocks if block.strip()]

data = []

for block in model_blocks:
    group_included = "Group_HR" in block

    r2_train = re.search(r"R2 train\s*=\s*([0-9.]+)", block)
    r2_test = re.search(r"R2 test\s*=\s*([0-9.]+)", block)
    run_time = re.search(r"Run completion time:\s*([0-9.]+)", block)

    # Extract parameters from OrderedDict
    param_match = re.search(r"Best Parameters:\s*OrderedDict\(\[(.*?)\]\)", block, re.DOTALL)
    param_dict = {}
    if param_match:
        param_text = param_match.group(1)
        param_dict = parse_ordered_dict(param_text)

    # Combine all data into a row
    row = {
        'Group Included': group_included,
        'R2 Train': float(r2_train.group(1)) if r2_train else None,
        'R2 Test': float(r2_test.group(1)) if r2_test else None,
        'Run Time': float(run_time.group(1)) if run_time else None,
    }

    if param_dict:
        row.update(param_dict)

    data.append(row)

# Step 3: Create the dataframe
df = pd.DataFrame(data)
print(df.head())