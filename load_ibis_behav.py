import pandas as pd

def load_ibis_behav(working_dir, filename):

    # read behavioral data from file
    behav_df = pd.read_csv(f'{working_dir}/IBIS_behav_dataframe_demographics_AnotB_Flanker_DCCS.csv',
                           usecols=lambda column: column != "Unnamed: 0")

    # remove columns not needed for analysis
    columns_to_remove = ['V12prefrontal_taskCandidate_Age', 'V24prefrontal_taskCandidate_Age',
                         'AB_Reversals_12_Percent', 'AB_Reversals_24_Percent', 'Age_SchoolAge']
    behav_df.drop(columns=columns_to_remove, inplace=True)

    # divide all scores by 100 to make data compatible with pcntoolkit
    score_columns = ['AB_12_Percent', 'AB_24_Percent', 'Flanker_Standard_Age_Corrected', 'DCCS_Standard_Age_Corrected']
    behav_df[score_columns] = behav_df[score_columns] / 100.0

    # convert gender to 0 (female) or male (1)
    behav_df.loc[behav_df['Sex'] == 'Female', 'Sex'] = 0
    behav_df.loc[behav_df['Sex'] == 'Male', 'Sex'] = 1

    return behav_df

