"""Produce bifurcation data comparing cases with and without identity (inter-behavioural dependence)

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import json
import numpy as np
from package.resources.utility import createFolder,produce_name_datetime,save_object,calc_pos_clusters_set_bandwidth
from package.resources.run import one_seed_culture_data_run
from package.generating_data.oneD_param_sweep_gen import (
    produce_param_list,
)

def main( 
    BASE_PARAMS_LOAD = "package/constants/base_params.json",
    no_samples = 10000,
    bandwidth = 0.01,
    param_min = 0.0,
    param_max = 100.0,
    reps = 500,

    ) -> str:

    property_varied = "confirmation_bias"

    ###FIRST RUN WITH IDENTITY (BEHAVIORAL INTERDEPENDANCE), ALPHA  = 1.0
    property_values_list = np.linspace(param_min,param_max, reps)

    f = open(BASE_PARAMS_LOAD)
    base_params = json.load(f)

    base_params["alpha_change"] = "dynamic_culturally_determined_weights"#Just to make sure

    params_list_identity = produce_param_list(base_params, property_values_list, property_varied)
    results_culture_lists_identity = one_seed_culture_data_run(params_list_identity)#list of lists lists [param set up, stochastic, cluster]

    #####################################################################
    ####NO IDENTITY, ALPHA  = 2.0

    base_params["alpha_change"] = "behavioural_independence"#Now change to behavioural independence
    params_list_no_identity = produce_param_list(base_params, property_values_list, property_varied)
    results_culture_lists_no_identity = one_seed_culture_data_run(params_list_no_identity)#list of lists lists [param set up, stochastic, cluster]

    ############################################################################

    root = "bifurcation_SINGLE_COMPARE_IDENTITY"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    createFolder(fileName)

    save_object(base_params, fileName + "/Data", "base_params")

    save_object(results_culture_lists_identity, fileName + "/Data", "results_culture_lists_identity")
    save_object(results_culture_lists_no_identity, fileName + "/Data", "results_culture_lists_no_identity")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_values_list, fileName + "/Data", "property_values_list")

    identity_space = np.linspace(0, 1,no_samples)

    cluster_pos_matrix_list_identity = [calc_pos_clusters_set_bandwidth(np.asarray(results_culture_lists_identity[i]),identity_space,bandwidth) for i in range(reps)] 
    cluster_pos_matrix_list_no_identity = [calc_pos_clusters_set_bandwidth(np.asarray(results_culture_lists_no_identity[i]),identity_space,bandwidth) for i in range(reps)] 
    
    save_object(cluster_pos_matrix_list_no_identity, fileName + "/Data", "cluster_pos_matrix_list_no_identity")
    save_object(cluster_pos_matrix_list_identity, fileName + "/Data", "cluster_pos_matrix_list_identity")
    save_object(identity_space, fileName + "/Data", "identity_space")
    save_object(bandwidth, fileName + "/Data", "bandwidth")

    return fileName

if __name__ == '__main__':
    fileName_Figure_5 = main(
    BASE_PARAMS_LOAD = "package/constants/base_params.json",
    param_min = 0.0,
    param_max = 100.0,
    reps = 500,
    bandwidth = 0.01
    )
