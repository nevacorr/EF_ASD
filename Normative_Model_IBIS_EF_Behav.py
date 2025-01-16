#####
# This program implements the bayesian linear regression normative modeling outlined by Rutherford et al.
# NatureProtocols 2022 (https://doi.org/10.1038/s41596-022-00696-5). Here the modeling is applied to
# behavioral data from IBIS children.
# Author: Neva M. Corrigan
######

import os
import matplotlib.pyplot as plt
from plot_and_compute_zdistributions import plot_and_compute_zcores_by_gender
from make_and_apply_normative_model import make_and_apply_normative_model

behav_var = 'Flanker and DCCS'
n_splits = 1   #Number of train/test splits
show_plots = 0          #set to 1 to show training and test data ymvs yhat and spline fit plots.
show_nsubject_plots = 0 #set to 1 to plot number of subjects used in analysis, for each age and gender
spline_order = 1        # order of spline to use for model
spline_knots = 2        # number of knots in spline to use in model
perform_train_test_split_precovid = 0 #flag indicating whether to split the training set (pre-COVID data) into train and validation data
data_dir = '/home/toddr/neva/PycharmProjects/data_dir'

datafile = 'genz_tract_profile_data/genzFA_tractProfiles_visit1.csv'

working_dir = os.getcwd()

roi_ids, Z_flanker, Z_dccs = make_and_apply_normative_model(behav_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                                                            data_dir, working_dir, datafile, n_splits)

plt.show(block=False)

tmp = Z_flanker.groupby(by=['participant_id'])
Z_flanker = Z_flanker.groupby(by=['participant_id']).mean().drop(columns=['split'])
Z_dccs = Z_dccs.groupby(by=['participant_id']).mean().drop(columns=['split'])

plot_and_compute_zcores_by_gender(Z_flanker, 'flanker', roi_ids, working_dir, n_splits)
Z_flanker.to_csv(f'{working_dir}/Z_flanker_{n_splits}_splits.csv')

roi_ids_md = roi_ids.copy()
roi_ids_md = [s.replace('flanker', 'dccs') for s in roi_ids_md]
plot_and_compute_zcores_by_gender(Z_dccs, 'dccs', roi_ids_md, working_dir, n_splits)
Z_dccs.to_csv(f'{working_dir}/Z_dccs_{n_splits}_splits.csv')

plt.show()

mystop=1