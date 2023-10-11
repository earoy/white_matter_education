
import numpy as np
import pandas as pd
import pickle as pkl

from bag_utils import *

###															 ###
### This script is meant to be passed into a slurm job array ###
### so that each model is trained in it's own batch jub		 ###
###			                                                 ###

if __name__ == "__main__":

	## number of each job array grabbed from command line arguments
	step = int(sys.argv[1])
	
	## Directory and file names for the AFQ and behavioral data
	## Overview on using these files to analyze the data can be 
	## found here: https://yeatmanlab.github.io/AFQ-Insight/auto_examples/demo_afq_dataset.html

	workdir = "/scratch/users/ethanroy/abcd_data"
	fn_nodes=op.join(workdir, "harmonized_nodes_two_obs.csv")
	fn_subjects=op.join(workdir, "subjects_two_obs_test.csv")

    target_cols = ['site_id_l','initial_age', 'log_inc_needs_RPP', 'bachelors', 'fes_youth_sum',
                   'sex', 'mean_pds', 'comc_phenx_mean', 'led_sch_seda_s_mn_avg_eb','subjectID_solo']

    ## Load ABCD data as AFQDataset
	abcd_dataset = AFQDataset.from_files(
		fn_nodes=fn_nodes,
		fn_subjects=fn_subjects,
		dwi_metrics=["dki_fa", "dki_md","dki_ad","dki_rd"],
		target_cols=target_cols,
		concat_subject_session=True
	)

	## Generate datasets

	train_sizes = [0.01,0.05,0.1,0.2,0.5,0.7] # propotions of sample used for training
	prop_datasets = dict()

	# Generate our datasets and populate a dictionary containing these datasets
	for prop in train_sizes:
		prop_datasets[prop] = generate_dataset_splits(abcd_dataset, train_prop=prop)
	
	## output names for our model objects
	output_names = ["bag_dicts/resnet_abcd_0.01.pkl","bag_dicts/resnet_abcd_0.05.pkl",
					"bag_dicts/resnet_abcd_0.10.pkl","bag_dicts/resnet_abcd_0.20.pkl",
					"bag_dicts/resnet_abcd_0.50.pkl","bag_dicts/resnet_abcd_0.70.pkl"]

	## Train a model for each training proportion as specificed by our
	## batch job numbers

	if step == 0:
		mod_dict = train_model(prop_datasets[0.01], normative_status=1)
	if step == 1:
		mod_dict = train_model(prop_datasets[0.05], normative_status=1)
	elif step == 2:
		mod_dict = train_model(prop_datasets[0.1], normative_status=1)
	elif step == 3:
		mod_dict = train_model(prop_datasets[0.2], normative_status=1)
	elif step == 4:
		mod_dict = train_model(prop_datasets[0.5], normative_status=1)
	elif step == 5:
		mod_dict = train_model(prop_datasets[0.7], normative_status=1)

	with open(output_names[step],'wb') as f:
	    pkl.dump(mod_dict, f)

