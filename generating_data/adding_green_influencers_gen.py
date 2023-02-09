"""
Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import numpy as np
from resources.utility import (
    createFolder,
    save_object,
    produce_name_datetime,
)
from resources.run import (
    multi_stochstic_emissions_run_all_individual,
)

def calc_new_K(K,N, N_green):
    new_K = (K*(N + N_green - 1))/(N - 1)
    return int(round(new_K))

def gen_atttiudes_list(mean_list, sum_a_b):
    init_attitudes_list = []
    for i in mean_list:
        a = i*sum_a_b
        b = sum_a_b - a
        init_attitudes_list.append([a,b])
    return init_attitudes_list


def main(
    green_N = 20,
    mean_list_min = 0.01,
    mean_list_max = 0.99,
    mean_list_reps = 200
    ) -> str: 

    base_params = {
        "save_timeseries_data": 0, 
        "degroot_aggregation": 1,
        "network_structure": "small_world",
        "alpha_change" : 1.0,
        "guilty_individuals": 0,
        "moral_licensing": 0,
        "immutable_green_fountains": 1,
        "additional_greens":1,
        "polarisation_test": 0,
        "total_time": 3000,
        "delta_t": 1.0,
        "phi_lower": 0.01,
        "phi_upper": 0.05,
        "compression_factor": 10,
        "seed_list": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],#[1,2,3],#
        "set_seed": 1,
        "N": 200,
        "M": 3,
        "K": 20,
        "prob_rewire": 0.1,
        "culture_momentum_real": 1000,
        "learning_error_scale": 0.02,
        "discount_factor": 0.95,
        "homophily": 0.95,
        "homophilly_rate" : 1,
        "confirmation_bias": 20,
        "a_attitude": 1,
        "b_attitude": 1,
        "a_threshold": 1,
        "b_threshold": 1,
        "action_observation_I": 0.0,
        "action_observation_S": 0.0,
        "green_N": 0,
        "guilty_individual_power": 0
    }
    base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])
    base_params_add_green = base_params.copy()

    ###############################################################
    
    mean_list = np.linspace(mean_list_min,mean_list_max, mean_list_reps)
    sum_a_b = 4#set the degree of polarisation? i think the more polarised the stronger the effect will be

    init_attitudes_list = gen_atttiudes_list(mean_list, sum_a_b)# GET THE LIST

    params_list_default = []

    for i in init_attitudes_list:
        #print("i",i)
        base_params["a_attitude"] = i[0]
        base_params["b_attitude"] = i[1]
        params_list_default.append(base_params.copy())

   
    base_params_add_green["green_N"] = green_N
    #base_params_add_green["N"] = base_params["N"] + green_N
    green_K = calc_new_K(base_params["K"],base_params["N"], green_N)
    base_params_add_green["K"] = green_K
    #print("green_K, N",green_K,base_params_add_green["N"], base_params_add_green["green_N"])

    params_list_add_green  = []
    for i in init_attitudes_list:
        #print("i",i)
        base_params_add_green["a_attitude"] = i[0]
        base_params_add_green["b_attitude"] = i[1]
        params_list_add_green.append(base_params_add_green.copy())

    #############################################################

    #fileName = produceName(params, params_name)
    root = "splitting_eco_warriors_distance_reps"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    ##############################################################################
    #CULTURED RUN
    emissions_list_default, emissions_id_list_individual_default = multi_stochstic_emissions_run_all_individual(params_list_default)
    emissions_list_add_green, emissions_id_list_individual_add_green = multi_stochstic_emissions_run_all_individual(params_list_add_green)

    #emissions_list_individual_add_green is a list or matrix? the entries are difference means and then for each stochastic run a list of indiviudal emissions [means, stochastic, indivdual emissions]
    #emissions_pos_matrix needs to be means then indivduals, so need to aggregate across the stochastic runs, for the differences between individuals (they are indifferet places but same initial values)

    #go through each stochastic run and sutract

    emissions_difference_lists = []
    for i in range(len(mean_list)):
        mean_row = []
        for k in range(base_params["N"]):
            person_row = []
            for j in range(len(base_params["seed_list"])):
                emissions_difference_stochastic = ((emissions_id_list_individual_add_green[i,j][k] -  emissions_id_list_individual_default[i,j][k])/emissions_id_list_individual_default[i,j][k])*100
                person_row.append(emissions_difference_stochastic)
            mean_row.append(np.mean(person_row))
        emissions_difference_lists.append(mean_row)
    
    emissions_difference_matrix = np.asarray(emissions_difference_lists)

    createFolder(fileName)
    save_object(mean_list, fileName + "/Data", "mean_list")
    save_object(sum_a_b , fileName + "/Data", "sum_a_b")
    save_object(emissions_list_default, fileName + "/Data", "emissions_list_default")
    save_object(emissions_list_add_green, fileName + "/Data", "emissions_list_add_green")
    save_object(emissions_id_list_individual_default, fileName + "/Data", "emissions_id_list_individual_default")
    save_object(emissions_id_list_individual_add_green, fileName + "/Data", "emissions_id_list_individual_add_green")
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(init_attitudes_list,fileName + "/Data", "init_attitudes_list")
    save_object(green_N, fileName + "/Data", "green_N")
    save_object(green_K, fileName + "/Data", "green_K")
    save_object(emissions_difference_matrix,fileName + "/Data", "emissions_difference_matrix")

    return fileName

