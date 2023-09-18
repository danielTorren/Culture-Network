""" Generate data comparing effect of green influencers

Created: 10/10/2022
"""

import numpy as np
import json
from package.resources.utility import (
    createFolder,
    save_object,
    produce_name_datetime,
)
from package.resources.run import (
    multi_stochstic_emissions_run_all_individual,
)

def calc_new_K(K,N, N_green):
    """Adjust K to keep constant network densitiy"""
    new_K = (K*(N + N_green - 1))/(N - 1)
    return int(round(new_K))

def gen_atttiudes_list(mean_list, sum_a_b):
    """From a list of mean distribution values calc the corresponding a and  b values for a given polarisation"""
    init_identity_list = []
    for i in mean_list:
        a = i*sum_a_b
        b = sum_a_b - a
        init_identity_list.append([a,b])
    return init_identity_list

def calc_attribute_percentage_change(carbon_emissions_not_influencer_no_green_no_identity,carbon_emissions_not_influencer_no_green_identity,carbon_emissions_not_influencer_green_no_identity,carbon_emissions_not_influencer_green_identity,mean_list,base_params):

    emissions_difference_stochastic_compare_green  = ((carbon_emissions_not_influencer_green_identity -  carbon_emissions_not_influencer_green_no_identity)/carbon_emissions_not_influencer_green_no_identity)*100#relative change in emissiosn between rusn with greens, where we divide by case of no identity
    emissions_difference_stochastic_compare_no_green  = ((carbon_emissions_not_influencer_no_green_identity -  carbon_emissions_not_influencer_no_green_no_identity)/carbon_emissions_not_influencer_no_green_no_identity)*100#relative change in emissiosn between rusn with no greens, where we divide by case of no identity
    emissions_difference_stochastic_compare_identity  = ((carbon_emissions_not_influencer_green_identity -  carbon_emissions_not_influencer_no_green_identity)/carbon_emissions_not_influencer_no_green_identity)*100#relative change in emissiosn between rusn with identity, where we divide by case of no greens
    emissions_difference_stochastic_compare_no_identity  = ((carbon_emissions_not_influencer_green_no_identity -  carbon_emissions_not_influencer_no_green_no_identity)/carbon_emissions_not_influencer_no_green_no_identity)*100#relative change in emissiosn between rusn with no identity, where we divide by case of no greens

    return emissions_difference_stochastic_compare_green,emissions_difference_stochastic_compare_no_green,emissions_difference_stochastic_compare_identity,emissions_difference_stochastic_compare_no_identity


def main(RUN = 1,BASE_PARAMS_LOAD = "package/constants/base_params_add_greens.json", param_vary_reps = 100,green_N = 20,sum_a_b = 2, confirmation_bias = 5):    
    
    if RUN:
        f = open(BASE_PARAMS_LOAD)
        base_params = json.load(f)
        
        ###############################################################
        
        mean_list = np.linspace(0.01,0.99, param_vary_reps)

        init_identity_list = gen_atttiudes_list(mean_list, sum_a_b)# GET THE LIST

        base_params["confirmation_bias"] = confirmation_bias
        ##########################################################################################
        #I NEED TO GENERATE THE PARAMS LIST FOR THE FOUR CASES, (NO CULTURE; NO GREEN), (CULTURE, NO GREEN),(NO CULTURE; GREEN) , (CULTURE; GREEN)
        # ADDING GREENS

        base_params_green_identity = base_params.copy()
        base_params_green_no_identity = base_params.copy()

        base_params_green_identity["green_N"] = green_N
        base_params_green_no_identity["green_N"] = green_N

        base_params_green_identity["green_N"] = green_N
        base_params_green_no_identity["green_N"] = green_N
        green_K = calc_new_K(base_params_green_identity["K"],base_params_green_identity["N"], green_N)
        base_params_green_identity["K"] = green_K
        base_params_green_no_identity["K"] = green_K

        params_list_green_identity = []
        base_params_green_identity["alpha_change"] = "dynamic_culturally_determined_weights"
        for i in init_identity_list:
            #print("i",i)
            base_params_green_identity["a_identity"] = i[0]
            base_params_green_identity["b_identity"] = i[1]
            params_list_green_identity.append(base_params_green_identity.copy())

        params_list_green_no_identity  = []
        base_params_green_no_identity["alpha_change"] = "behavioural_independence"
        for i in init_identity_list:
            #print("i",i)
            base_params_green_no_identity["a_identity"] = i[0]
            base_params_green_no_identity["b_identity"] = i[1]
            params_list_green_no_identity.append(base_params_green_no_identity.copy())


        ################################################################
        #NO GREENS
        

        base_params_no_green_identity = base_params.copy()
        base_params_no_green_no_identity = base_params.copy()
        
        params_list_no_green_identity = []
        base_params_no_green_identity["alpha_change"] = "dynamic_culturally_determined_weights"
        for i in init_identity_list:
            #print("i",i)
            base_params_no_green_identity["a_identity"] = i[0]
            base_params_no_green_identity["b_identity"] = i[1]
            params_list_no_green_identity.append(base_params_no_green_identity.copy())

        params_list_no_green_no_identity  = []
        base_params_no_green_no_identity["alpha_change"] = "behavioural_independence"
        for i in init_identity_list:
            #print("i",i)
            base_params_no_green_no_identity["a_identity"] = i[0]
            base_params_no_green_no_identity["b_identity"] = i[1]
            params_list_no_green_no_identity.append(base_params_no_green_no_identity.copy())

        #############################################################

        #fileName = produceName(params, params_name)
        root = "green_influencers_identity_four_alt"
        fileName = produce_name_datetime(root)
        print("fileName:", fileName)

        ##############################################################################
        #DO THE RUNS
        #GREENS
        emissions_list_green_no_identity, carbon_emissions_not_influencer_green_no_identity  = multi_stochstic_emissions_run_all_individual(params_list_green_no_identity)        
        emissions_list_green_identity, carbon_emissions_not_influencer_green_identity = multi_stochstic_emissions_run_all_individual(params_list_green_identity)
        #NO GREENS
        emissions_list_no_green_no_identity, carbon_emissions_not_influencer_no_green_no_identity = multi_stochstic_emissions_run_all_individual(params_list_no_green_no_identity)
        emissions_list_no_green_identity, carbon_emissions_not_influencer_no_green_identity = multi_stochstic_emissions_run_all_individual(params_list_no_green_identity)

        ####################################################################################
        emissions_difference_matrix_compare_green,emissions_difference_matrix_compare_no_green,emissions_difference_matrix_compare_identity,emissions_difference_matrix_compare_no_identity = calc_attribute_percentage_change(carbon_emissions_not_influencer_no_green_no_identity,carbon_emissions_not_influencer_no_green_identity,carbon_emissions_not_influencer_green_no_identity,carbon_emissions_not_influencer_green_identity,mean_list,base_params)
        createFolder(fileName)

        save_object(emissions_list_no_green_no_identity, fileName + "/Data", "emissions_list_no_green_no_identity")
        save_object(emissions_list_no_green_identity, fileName + "/Data", "emissions_list_no_green_identity")
        save_object(emissions_list_green_no_identity, fileName + "/Data", "emissions_list_green_no_identity")
        save_object(emissions_list_green_identity, fileName + "/Data", "emissions_list_green_identity")

        save_object(carbon_emissions_not_influencer_no_green_no_identity, fileName + "/Data", "carbon_emissions_not_influencer_no_green_no_identity")
        save_object(carbon_emissions_not_influencer_no_green_identity, fileName + "/Data", "carbon_emissions_not_influencer_no_green_identity")
        save_object(carbon_emissions_not_influencer_green_no_identity, fileName + "/Data", "carbon_emissions_not_influencer_green_no_identity")
        save_object(carbon_emissions_not_influencer_green_identity, fileName + "/Data", "carbon_emissions_not_influencer_green_identity")

        save_object(emissions_difference_matrix_compare_green,fileName + "/Data", "emissions_difference_matrix_compare_green")
        save_object(emissions_difference_matrix_compare_no_green,fileName + "/Data", "emissions_difference_matrix_compare_no_green")
        save_object(emissions_difference_matrix_compare_identity,fileName + "/Data", "emissions_difference_matrix_compare_identity")
        save_object(emissions_difference_matrix_compare_no_identity,fileName + "/Data", "emissions_difference_matrix_compare_no_identity")

        save_object(mean_list, fileName + "/Data", "mean_list")
        save_object(sum_a_b , fileName + "/Data", "sum_a_b")
        save_object(base_params, fileName + "/Data", "base_params")
        save_object(init_identity_list,fileName + "/Data", "init_identity_list")
        save_object(green_N, fileName + "/Data", "green_N")
        save_object(green_K, fileName + "/Data", "green_K")   
    return fileName

if __name__ == '__main__':

    fileName_Figure_9_low = main(RUN = 1, BASE_PARAMS_LOAD = "package/constants/base_params_add_greens.json", param_vary_reps = 200, green_N = 20, sum_a_b = 6, confirmation_bias = 5)
    fileName_Figure_9_high = main(RUN = 1, BASE_PARAMS_LOAD = "package/constants/base_params_add_greens.json", param_vary_reps = 200, green_N = 20, sum_a_b = 6, confirmation_bias = 20)

