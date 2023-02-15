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

def produce_param_list_polarisation(params: dict, property_list: list, property: str) -> list[dict]:
    params_list = []
    for i in property_list:
        params["a_attitude"] = i
        params["b_attitude"] = i
        params_list.append(
            params.copy()
        )  # have to make a copy so that it actually appends a new dict and not just the location of the params dict
    return params_list

def main( 
    BASE_PARAMS_LOAD = "package/constants/base_params_consensus.json",
    param_min = 0.0,
    param_max = 100.0,
    reps = 500,
    ) -> str:

    property_varied = "a_attitude"

    ###FIRST RUN WITH IDENTITY (BEHAVIORAL INTERDEPENDANCE), ALPHA  = 1.0
    property_values_list = np.linspace(param_min,param_max, reps)

    f = open(BASE_PARAMS_LOAD)
    base_params = json.load(f)

    base_params["alpha_change"] = "dynamic_culturally_determined_weights"#Just to make sure

    params_list_identity = produce_param_list_polarisation(base_params, property_values_list, property_varied)
    results_culture_lists_identity = one_seed_culture_data_run(params_list_identity)#list of lists lists [param set up, stochastic, cluster]

    #####################################################################
    ####NO IDENTITY, ALPHA  = 2.0

    base_params["alpha_change"] = "behavioural_independence"#Now change to behavioural independence
    params_list_no_identity = produce_param_list_polarisation(base_params, property_values_list, property_varied)
    results_culture_lists_no_identity = one_seed_culture_data_run(params_list_no_identity)#list of lists lists [param set up, stochastic, cluster]

    ############################################################################

    root = "consensus_formation"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    createFolder(fileName)

    save_object(base_params, fileName + "/Data", "base_params")

    save_object(results_culture_lists_identity, fileName + "/Data", "results_culture_lists_identity")
    save_object(results_culture_lists_no_identity, fileName + "/Data", "results_culture_lists_no_identity")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(property_values_list, fileName + "/Data", "property_values_list")


    ####################################################
    #CALC THE VARIANCE OF A
    #print("results_culture_lists_identity",results_culture_lists_identity,results_culture_lists_identity.shape)
    var_identity  = np.var(results_culture_lists_identity, axis=1)
    var_no_identity  = np.var(results_culture_lists_no_identity, axis=1)
    print("var_identity", var_identity)
    print("var_no_identity", var_no_identity)
    
    save_object(var_identity, fileName + "/Data", "var_identity")
    save_object(var_no_identity, fileName + "/Data", "var_no_identity")

    return fileName

if __name__ == '__main__':
    fileName_Figure_5 = main(
    BASE_PARAMS_LOAD = "package/constants/base_params_consensus.json",
    param_min = 0.05,
    param_max = 1,
    reps = 100,
    )
