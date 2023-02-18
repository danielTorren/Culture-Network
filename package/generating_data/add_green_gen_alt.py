import numpy as np
import json
import matplotlib.pyplot as plt
from package.resources.utility import (
    createFolder,
    save_object,
    produce_name_datetime,
    load_object
)
from package.resources.run import (
    multi_stochstic_emissions_run_all_individual,
)
from package.resources.plot import (
    plot_four,
    plot_four_two
)

def gen_atttiudes_list(mean_list, sum_a_b):
    init_attitudes_list = []
    for i in mean_list:
        a = i*sum_a_b
        b = sum_a_b - a
        init_attitudes_list.append([a,b])
    return init_attitudes_list

def calc_attribute_percentage_change(carbon_emissions_not_influencer_no_green_no_culture,carbon_emissions_not_influencer_no_green_culture,carbon_emissions_not_influencer_green_no_culture,carbon_emissions_not_influencer_green_culture,mean_list,base_params):

    emissions_difference_stochastic_compare_green  = ((carbon_emissions_not_influencer_green_culture -  carbon_emissions_not_influencer_green_no_culture)/carbon_emissions_not_influencer_green_no_culture)*100#relative change in emissiosn between rusn with greens, where we divide by case of no culture
    emissions_difference_stochastic_compare_no_green  = ((carbon_emissions_not_influencer_no_green_culture -  carbon_emissions_not_influencer_no_green_no_culture)/carbon_emissions_not_influencer_no_green_no_culture)*100#relative change in emissiosn between rusn with no greens, where we divide by case of no culture
    emissions_difference_stochastic_compare_culture  = ((carbon_emissions_not_influencer_green_culture -  carbon_emissions_not_influencer_no_green_culture)/carbon_emissions_not_influencer_no_green_culture)*100#relative change in emissiosn between rusn with culture, where we divide by case of no greens
    emissions_difference_stochastic_compare_no_culture  = ((carbon_emissions_not_influencer_green_no_culture -  carbon_emissions_not_influencer_no_green_no_culture)/carbon_emissions_not_influencer_no_green_no_culture)*100#relative change in emissiosn between rusn with no culture, where we divide by case of no greens

    return emissions_difference_stochastic_compare_green,emissions_difference_stochastic_compare_no_green,emissions_difference_stochastic_compare_culture,emissions_difference_stochastic_compare_no_culture


def main(RUN = 1, fileName = "results/green_influencers_culture_four_alt_19_37_22__16_02_2023",BASE_PARAMS_LOAD = "package/constants/base_params_alt.json",dpi_save = 1200, param_vary_reps = 100,green_N = 20,sum_a_b = 2, confirmation_bias = 5):    
    
    if RUN:
        f = open(BASE_PARAMS_LOAD)
        base_params = json.load(f)
        
        ###############################################################
        
        mean_list = np.linspace(0.01,0.99, param_vary_reps)
        #set the degree of polarisation? i think the more polarised the stronger the effect will be

        init_attitudes_list = gen_atttiudes_list(mean_list, sum_a_b)# GET THE LIST


        base_params["confirmation_bias"] = confirmation_bias
        ##########################################################################################
        ####################################################################
        # NOW DO ADDING GREENS
        
        base_params_green_culture = base_params.copy()
        base_params_green_no_culture = base_params.copy()

        base_params_green_culture["green_N"] = green_N
        base_params_green_no_culture["green_N"] = green_N

        params_list_green_culture = []
        base_params_green_culture["alpha_change"] = "dynamic_culturally_determined_weights"
        for i in init_attitudes_list:
            #print("i",i)
            base_params_green_culture["a_attitude"] = i[0]
            base_params_green_culture["b_attitude"] = i[1]
            params_list_green_culture.append(base_params_green_culture.copy())

        params_list_green_no_culture  = []
        base_params_green_no_culture["alpha_change"] = "behavioural_independence"
        for i in init_attitudes_list:
            #print("i",i)
            base_params_green_no_culture["a_attitude"] = i[0]
            base_params_green_no_culture["b_attitude"] = i[1]
            params_list_green_no_culture.append(base_params_green_no_culture.copy())


        ################################################################
        #I NEED TO GENERATE THE PARAMS LIST FOR THE FOUR CASES, (NO CULUTER; NO GREEN), (CULTURE, NO GREEN),(NO CULTURE; GREEN) , (CULTURE; GREEN)
        # START WITH THE NO GREEN RUNS

        base_params_no_green_culture = base_params.copy()
        base_params_no_green_no_culture = base_params.copy()
        
        params_list_no_green_culture = []
        base_params_no_green_culture["alpha_change"] = "dynamic_culturally_determined_weights"
        for i in init_attitudes_list:
            #print("i",i)
            base_params_no_green_culture["a_attitude"] = i[0]
            base_params_no_green_culture["b_attitude"] = i[1]
            params_list_no_green_culture.append(base_params_no_green_culture.copy())

        params_list_no_green_no_culture  = []
        base_params_no_green_no_culture["alpha_change"] = "behavioural_independence"
        for i in init_attitudes_list:
            #print("i",i)
            base_params_no_green_no_culture["a_attitude"] = i[0]
            base_params_no_green_no_culture["b_attitude"] = i[1]
            params_list_no_green_no_culture.append(base_params_no_green_no_culture.copy())

        #############################################################

        #fileName = produceName(params, params_name)
        root = "green_influencers_culture_four_alt"
        fileName = produce_name_datetime(root)
        print("fileName:", fileName)

        ##############################################################################
        #DO THE RUNS
        #GREENS
        emissions_list_green_no_culture, carbon_emissions_not_influencer_green_no_culture  = multi_stochstic_emissions_run_all_individual(params_list_green_no_culture)        
        emissions_list_green_culture, carbon_emissions_not_influencer_green_culture = multi_stochstic_emissions_run_all_individual(params_list_green_culture)
        #NO GREENS
        emissions_list_no_green_no_culture, carbon_emissions_not_influencer_no_green_no_culture = multi_stochstic_emissions_run_all_individual(params_list_no_green_no_culture)
        emissions_list_no_green_culture, carbon_emissions_not_influencer_no_green_culture = multi_stochstic_emissions_run_all_individual(params_list_no_green_culture)

        print("DONE,",emissions_list_green_no_culture.shape,  carbon_emissions_not_influencer_green_no_culture.shape )

        ####################################################################################
        emissions_difference_matrix_compare_green,emissions_difference_matrix_compare_no_green,emissions_difference_matrix_compare_culture,emissions_difference_matrix_compare_no_culture = calc_attribute_percentage_change(carbon_emissions_not_influencer_no_green_no_culture,carbon_emissions_not_influencer_no_green_culture,carbon_emissions_not_influencer_green_no_culture,carbon_emissions_not_influencer_green_culture,mean_list,base_params)
        print("LETS GET SAVIN")
        createFolder(fileName)

        save_object(emissions_list_no_green_no_culture, fileName + "/Data", "emissions_list_no_green_no_culture")
        save_object(emissions_list_no_green_culture, fileName + "/Data", "emissions_list_no_green_culture")
        save_object(emissions_list_green_no_culture, fileName + "/Data", "emissions_list_green_no_culture")
        save_object(emissions_list_green_culture, fileName + "/Data", "emissions_list_green_culture")

        save_object(carbon_emissions_not_influencer_no_green_no_culture, fileName + "/Data", "carbon_emissions_not_influencer_no_green_no_culture")
        save_object(carbon_emissions_not_influencer_no_green_culture, fileName + "/Data", "carbon_emissions_not_influencer_no_green_culture")
        save_object(carbon_emissions_not_influencer_green_no_culture, fileName + "/Data", "carbon_emissions_not_influencer_green_no_culture")
        save_object(carbon_emissions_not_influencer_green_culture, fileName + "/Data", "carbon_emissions_not_influencer_green_culture")

        save_object(emissions_difference_matrix_compare_green,fileName + "/Data", "emissions_difference_matrix_compare_green")
        save_object(emissions_difference_matrix_compare_no_green,fileName + "/Data", "emissions_difference_matrix_compare_no_green")
        save_object(emissions_difference_matrix_compare_culture,fileName + "/Data", "emissions_difference_matrix_compare_culture")
        save_object(emissions_difference_matrix_compare_no_culture,fileName + "/Data", "emissions_difference_matrix_compare_no_culture")

        save_object(mean_list, fileName + "/Data", "mean_list")
        save_object(sum_a_b , fileName + "/Data", "sum_a_b")
        save_object(base_params, fileName + "/Data", "base_params")
        save_object(init_attitudes_list,fileName + "/Data", "init_attitudes_list")
        save_object(green_N, fileName + "/Data", "green_N")
        
    else:
        emissions_list_no_green_no_culture = load_object( fileName + "/Data", "emissions_list_no_green_no_culture")
        emissions_list_no_green_culture = load_object( fileName + "/Data", "emissions_list_no_green_culture")
        emissions_list_green_no_culture = load_object( fileName + "/Data", "emissions_list_green_no_culture")
        emissions_list_green_culture = load_object( fileName + "/Data", "emissions_list_green_culture")

        carbon_emissions_not_influencer_no_green_no_culture = load_object( fileName + "/Data", "carbon_emissions_not_influencer_no_green_no_culture")
        carbon_emissions_not_influencer_no_green_culture = load_object( fileName + "/Data", "carbon_emissions_not_influencer_no_green_culture")
        carbon_emissions_not_influencer_green_no_culture = load_object( fileName + "/Data", "carbon_emissions_not_influencer_green_no_culture")
        carbon_emissions_not_influencer_green_culture = load_object( fileName + "/Data", "carbon_emissions_not_influencer_green_culture")

        emissions_difference_matrix_compare_green = load_object(fileName + "/Data", "emissions_difference_matrix_compare_green")
        emissions_difference_matrix_compare_no_green = load_object(fileName + "/Data", "emissions_difference_matrix_compare_no_green")
        emissions_difference_matrix_compare_culture = load_object(fileName + "/Data", "emissions_difference_matrix_compare_culture")
        emissions_difference_matrix_compare_no_culture = load_object(fileName + "/Data", "emissions_difference_matrix_compare_no_culture")

        mean_list = load_object( fileName + "/Data", "mean_list")
        sum_a_b  = load_object( fileName + "/Data", "sum_a_b")
        base_params = load_object( fileName + "/Data", "base_params")
        init_attitudes_list = load_object(fileName + "/Data", "init_attitudes_list")
        green_N = load_object( fileName + "/Data", "green_N")

    #plot_four(fileName, np.asarray(emissions_list_no_green_no_culture), np.asarray(emissions_list_no_green_culture), np.asarray(emissions_list_green_no_culture), np.asarray(emissions_list_green_culture), base_params["confirmation_bias"],mean_list, dpi_save)
    plot_four(fileName,carbon_emissions_not_influencer_no_green_no_culture, carbon_emissions_not_influencer_no_green_culture, carbon_emissions_not_influencer_green_no_culture, carbon_emissions_not_influencer_green_culture, base_params["confirmation_bias"],mean_list, dpi_save)
    plot_four_two(fileName, emissions_difference_matrix_compare_green, emissions_difference_matrix_compare_no_green, emissions_difference_matrix_compare_culture, emissions_difference_matrix_compare_no_culture, mean_list, dpi_save)
    plt.show()  

    return fileName

if __name__ == '__main__':

    main(RUN = 1, fileName = "results/green_influencers_culture_four_alt_19_37_22__16_02_2023",BASE_PARAMS_LOAD = "package/constants/base_params_alt.json",dpi_save = 1200, param_vary_reps = 100, green_N = 20, sum_a_b = 6, confirmation_bias = 5)