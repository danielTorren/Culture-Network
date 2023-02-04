"""Produce bifurcation plots

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import json
import matplotlib.pyplot as plt
import numpy as np
from resources.utility import createFolder,produce_name_datetime,save_object,calc_pos_clusters_set_bandwidth
from resources.run import culture_data_run,one_seed_culture_data_run
from oneD_param_sweep_gen import (
    produce_param_list,
)

def main( 
    RUN_NAME = "SINGLE"
    ) -> str:

    if RUN_NAME == "SINGLE":

        ###FIRST RUN WITH IDENTITY (BEHAVIORAL INTERDEPENDANCE), ALPHA  = 1.0
        property_varied_identity = "confirmation_bias"
        property_varied_title_identity = "Confirmation bias $\theta$"
        param_min_identity = 0.0
        param_max_identity = 100.0  # 50.0
        reps_identity = 500
        title_list_identity = ["Bifurcation"]
        #title_list = [r"Confirmation bias $\theta$ = 1.0", r"Confirmation bias $\theta$ = 10.0", r"Confirmation bias $\theta$ = 25.0", r"Confirmation bias $\theta$ = 50.0"]
        property_values_list_identity = np.linspace(param_min_identity,param_max_identity, reps_identity)
        print("property_values_list_identity ", property_values_list_identity )

        f = open("constants/base_params.json")
        base_params = json.load(f)
        base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])
        base_params["alpha_change"] = 1.0

        params_list_identity = produce_param_list(base_params, property_values_list_identity, property_varied_identity)
        results_culture_lists_identity = one_seed_culture_data_run(params_list_identity)#list of lists lists [param set up, stochastic, cluster]

        #####################################################################
        ####NO IDENTITY, ALPHA  = 2.0
        property_varied_no_identity = "confirmation_bias"
        property_varied_title_no_identity = "Confirmation bias $\theta$"
        param_min_no_identity = 0.0
        param_max_no_identity = 100.0  # 50.0
        reps_no_identity = 500
        title_list_no_identity = ["Bifurcation"]
        #title_list = [r"Confirmation bias $\theta$ = 1.0", r"Confirmation bias $\theta$ = 10.0", r"Confirmation bias $\theta$ = 25.0", r"Confirmation bias $\theta$ = 50.0"]
        property_values_list_no_identity = np.linspace(param_min_no_identity,param_max_no_identity, reps_no_identity)
        print("property_values_list_no_identity ", property_values_list_no_identity )

        base_params["alpha_change"] = 2.0

        params_list_no_identity = produce_param_list(base_params, property_values_list_no_identity, property_varied_no_identity)
        results_culture_lists_no_identity = one_seed_culture_data_run(params_list_no_identity)#list of lists lists [param set up, stochastic, cluster]

        ############################################################################

        root = "bifurcation_SINGLE_COMPARE_IDENTITY"
        fileName = produce_name_datetime(root)
        print("fileName: ", fileName)

        createFolder(fileName)

        save_object(base_params, fileName + "/Data", "base_params")

        save_object(results_culture_lists_identity, fileName + "/Data", "results_culture_lists_identity")
        save_object(property_varied_identity, fileName + "/Data", "property_varied_identity")
        save_object(property_varied_title_identity, fileName + "/Data", "property_varied_title_identity")
        save_object(param_min_identity, fileName + "/Data", "param_min_identity")
        save_object(param_max_identity, fileName + "/Data", "param_max_identity")
        save_object(title_list_identity, fileName + "/Data", "title_list_identity")
        save_object(property_values_list_identity, fileName + "/Data", "property_values_list_identity")

        save_object(results_culture_lists_no_identity, fileName + "/Data", "results_culture_lists_no_identity")
        save_object(property_varied_no_identity, fileName + "/Data", "property_varied_no_identity")
        save_object(property_varied_title_no_identity, fileName + "/Data", "property_varied_title_no_identity")
        save_object(param_min_no_identity, fileName + "/Data", "param_min_no_identity")
        save_object(param_max_no_identity, fileName + "/Data", "param_max_no_identity")
        save_object(title_list_no_identity, fileName + "/Data", "title_list_no_identity")
        save_object(property_values_list_no_identity, fileName + "/Data", "property_values_list_no_identity")
        ###WORKING
        N = base_params["N"]

        no_samples = 10000
        identity_space = np.linspace(0, 1,no_samples)
        bandwidth = 0.01

        cluster_pos_matrix_list_identity = [calc_pos_clusters_set_bandwidth(np.asarray(results_culture_lists_identity[i]),identity_space,bandwidth) for i in range(len(property_values_list_identity))] 
        cluster_pos_matrix_list_no_identity = [calc_pos_clusters_set_bandwidth(np.asarray(results_culture_lists_no_identity[i]),identity_space,bandwidth) for i in range(len(property_values_list_no_identity))] 
        
        save_object(cluster_pos_matrix_list_no_identity, fileName + "/Data", "cluster_pos_matrix_list_no_identity")
        save_object(cluster_pos_matrix_list_identity, fileName + "/Data", "cluster_pos_matrix_list_identity")

        save_object(identity_space, fileName + "/Data", "identity_space")
        save_object(bandwidth, fileName + "/Data", "bandwidth")

    elif RUN_NAME == "MULTI":
        ############################

        seed_list_set = list(range(20))
        print(seed_list_set)
        property_varied = "confirmation_bias"
        property_varied_title = "Confirmation bias $\theta$"
        param_min = -10.0
        param_max = 100.0  # 50.0
        reps = 220
        title_list = ["Bifurcation"]
        #title_list = [r"Confirmation bias $\theta$ = 1.0", r"Confirmation bias $\theta$ = 10.0", r"Confirmation bias $\theta$ = 25.0", r"Confirmation bias $\theta$ = 50.0"]
        property_values_list = np.linspace(param_min,param_max, reps)
        print("property_values_list ", property_values_list )

        f = open("constants/base_params.json")
        base_params = json.load(f)
        base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])
        base_params["seed_list"] = seed_list_set

        root = "bifurcation_one_param"
        fileName = produce_name_datetime(root)
        print("fileName: ", fileName)

        params_list = produce_param_list(base_params, property_values_list, property_varied)

        results_culture_lists = culture_data_run(params_list)#list of lists lists [param set up, stochastic, cluster]

        createFolder(fileName)

        save_object(results_culture_lists, fileName + "/Data", "results_culture_lists")
        save_object(base_params, fileName + "/Data", "base_params")
        save_object(property_varied, fileName + "/Data", "property_varied")
        save_object(property_varied_title, fileName + "/Data", "property_varied_title")
        save_object(param_min, fileName + "/Data", "param_min")
        save_object(param_max, fileName + "/Data", "param_max")
        save_object(title_list, fileName + "/Data", "title_list")
        save_object(property_values_list, fileName + "/Data", "property_values_list")

        ###WORKING
        N = base_params["N"]
        seed_list = base_params["seed_list"]

        no_samples = 10000
        identity_space = np.linspace(0, 1,no_samples)
        bandwidth = 0.01

        cluster_pos_matrix_list = [[calc_pos_clusters_set_bandwidth(np.asarray(results_culture_lists[i][j]),identity_space,bandwidth) for i in range(len(property_values_list))] for j in range(len(seed_list))]
        #print("cluster_pos_matrix_list",cluster_pos_matrix_list,len(cluster_pos_matrix_list))
        # each entry is a matrix of position of identity clusters, corresponding to one stochastic run
        
        save_object(cluster_pos_matrix_list, fileName + "/Data", "cluster_pos_matrix_list")
        save_object(identity_space, fileName + "/Data", "identity_space")
        save_object(bandwidth, fileName + "/Data", "bandwidth")

    return fileName
