
import pandas as pd
import os

datadir = '/home/toddr/neva/PycharmProjects/data_dir/IBIS_Behav_Brain'
# Load Executive Function data

anotb = pd.read_csv(os.path.join(datadir, 'AnotB_clean.csv'))
brief1 = pd.read_csv(os.path.join(datadir, 'BRIEF1_UNC.csv'))
brief2 = pd.read_csv(os.path.join(datadir, 'BRIEF-2_7-1-24_data-2024-07-01T19_35_29.390Z.csv'))
dx = pd.read_csv(os.path.join(datadir, 'DSM_7-1-24_data-2024-07-01T21_05_03.559Z.csv'))
dx2 = pd.read_csv(os.path.join(datadir, 'New-11-22-21_data-2021-11-23T07_39_34.455Z.csv'))
nihtoolbox = pd.read_csv(os.path.join(datadir, 'NIH Toolbox_7-1-24_data-2024-07-01T19_40_36.204Z.csv'))

anotb_cols_to_remove = ['DCCID', 'V06demographicProject', 'V06demographicsRisk', 'V06demographicsSite', 'V06demographicsStatus', 'V12demographicsProject']
brief2_cols_to_keep = ['Identifiers', 'VSA BRIEF2_Parent,Candidate_Age', 'VSA-CVD BRIEF2_Parent,Candidate_Age']
brief2_cols_to_keep.append([col for col in brief2.columns.tolist() if 'raw_score' in col or 'T_score' in col])
brief2_cols_to_keep.remove('VSA BRIEF2_Parent,T_score')
brief2_cols_to_keep.remove('VSA-CVD BRIEF2_Parent,T_score')





mystop=1

