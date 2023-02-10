""" Generate data comparing effect of green influencers
Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import numpy as np
import json
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
    BASE_PARAMS_LOAD = "constants/base_params_add_greens.json",
    green_N = 20,
    mean_list_min = 0.01,
    mean_list_max = 0.99,
    mean_list_reps = 200,
    sum_a_b = 4,
    confirmation_bias = 5
    ) -> str: 

    f = open(BASE_PARAMS_LOAD)
    base_params = json.load(f)
    base_params["confirmation_bias"] = confirmation_bias

    base_params_add_green = base_params.copy()

    ###############################################################
    mean_list = np.linspace(mean_list_min,mean_list_max, mean_list_reps)

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


    params_list_add_green  = []
    for i in init_attitudes_list:

        base_params_add_green["a_attitude"] = i[0]
        base_params_add_green["b_attitude"] = i[1]
        params_list_add_green.append(base_params_add_green.copy())

    #############################################################

    root = "splitting_eco_warriors_distance_reps"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    ##############################################################################
    #CULTURED RUN
    emissions_list_default, emissions_id_list_individual_default = multi_stochstic_emissions_run_all_individual(params_list_default)
    emissions_list_add_green, emissions_id_list_individual_add_green = multi_stochstic_emissions_run_all_individual(params_list_add_green)

    #go through each stochastic run and subtract

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

