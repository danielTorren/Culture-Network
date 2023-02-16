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
    plot_four_two,
    plot_four_two_attitude,
    plot_id_change,
    plot_id_change_cases
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

def calc_id_attribute_change(emissions_id_list_no_green_no_culture,emissions_id_list_no_green_culture,emissions_id_list_green_no_culture,emissions_id_list_green_culture,mean_list,base_params):
    emissions_difference_lists_compare_green = []
    emissions_difference_lists_compare_no_green = []
    emissions_difference_lists_compare_culture = []
    emissions_difference_lists_compare_no_culture = []
    for i in range(len(mean_list)):
        mean_row_compare_green = []
        mean_row_compare_no_green = []
        mean_row_compare_culture = []
        mean_row_compare_no_culture = []
        for k in range(base_params["N"]):
            person_row_compare_green = []
            person_row_compare_no_green = []
            person_row_compare_culture = []
            person_row_compare_no_culture = []
            for j in range(len(base_params["seed_list"])):
                emissions_difference_stochastic_compare_green  = ((emissions_id_list_green_culture[i,j][k] -  emissions_id_list_green_no_culture[i,j][k])/emissions_id_list_green_no_culture[i,j][k])*100#relative change in emissiosn between rusn with greens, where we divide by case of no culture
                emissions_difference_stochastic_compare_no_green  = ((emissions_id_list_no_green_culture[i,j][k] -  emissions_id_list_no_green_no_culture[i,j][k])/emissions_id_list_no_green_no_culture[i,j][k])*100#relative change in emissiosn between rusn with no greens, where we divide by case of no culture
                emissions_difference_stochastic_compare_culture  = ((emissions_id_list_green_culture[i,j][k] -  emissions_id_list_no_green_culture[i,j][k])/emissions_id_list_no_green_culture[i,j][k])*100#relative change in emissiosn between rusn with culture, where we divide by case of no greens
                emissions_difference_stochastic_compare_no_culture  = ((emissions_id_list_green_no_culture[i,j][k] -  emissions_id_list_no_green_no_culture[i,j][k])/emissions_id_list_no_green_no_culture[i,j][k])*100#relative change in emissiosn between rusn with no culture, where we divide by case of no greens

                person_row_compare_green.append(emissions_difference_stochastic_compare_green)
                person_row_compare_no_green.append(emissions_difference_stochastic_compare_no_green)
                person_row_compare_culture.append(emissions_difference_stochastic_compare_culture)
                person_row_compare_no_culture.append(emissions_difference_stochastic_compare_no_culture)

            mean_row_compare_green.append(np.mean(person_row_compare_green))
            mean_row_compare_no_green.append(np.mean(person_row_compare_no_green))
            mean_row_compare_culture.append(np.mean(person_row_compare_culture))
            mean_row_compare_no_culture.append(np.mean(person_row_compare_no_culture))
        emissions_difference_lists_compare_green.append(mean_row_compare_green)
        emissions_difference_lists_compare_no_green.append(mean_row_compare_no_green)
        emissions_difference_lists_compare_culture.append(mean_row_compare_culture)
        emissions_difference_lists_compare_no_culture.append(mean_row_compare_no_culture)
    
    emissions_difference_matrix_compare_green = np.asarray(emissions_difference_lists_compare_green)
    emissions_difference_matrix_compare_no_green = np.asarray(emissions_difference_lists_compare_no_green)
    emissions_difference_matrix_compare_culture = np.asarray(emissions_difference_lists_compare_culture)
    emissions_difference_matrix_compare_no_culture = np.asarray(emissions_difference_lists_compare_no_culture)

    return emissions_difference_matrix_compare_green,emissions_difference_matrix_compare_no_green,emissions_difference_matrix_compare_culture,emissions_difference_matrix_compare_no_culture


def calc_id_attribute_change_start_finish(final_id_no_green_no_culture, final_id_no_green_culture,final_id_green_no_culture,final_id_green_culture,initial_id_no_green_no_culture, initial_id_no_green_culture,initial_id_green_no_culture,initial_id_green_culture,mean_list,base_params):
    
    attribute_difference_lists_no_green_no_culture = []
    attribute_difference_lists_green_no_culture = []
    attribute_difference_lists_no_green_culture = []
    attribute_difference_lists_green_culture = []

    for i in range(len(mean_list)):
        experiment_row_no_green_no_culture = []
        experiment_row_green_no_culture = []
        experiment_row_no_green_culture = []
        experiment_row_green_culture = []
        for k in range(base_params["N"]):
            person_row_no_green_no_culture = []
            person_row_green_no_culture = []
            person_row_no_green_culture = []
            person_row_green_culture = []
            for j in range(len(base_params["seed_list"])):
                attribute_difference_stochastic_no_green_no_culture  = [initial_id_no_green_no_culture[i,j][k],((final_id_no_green_no_culture[i,j][k] -  initial_id_no_green_no_culture[i,j][k])/initial_id_no_green_no_culture[i,j][k])*100]
                attribute_difference_stochastic_green_no_culture  = [initial_id_green_no_culture[i,j][k],((final_id_green_no_culture[i,j][k] -  initial_id_green_no_culture[i,j][k])/initial_id_green_no_culture[i,j][k])*100]
                attribute_difference_stochastic_no_green_culture  = [initial_id_no_green_culture[i,j][k],((final_id_no_green_culture[i,j][k] -  initial_id_no_green_culture[i,j][k])/initial_id_no_green_culture[i,j][k])*100]
                attribute_difference_stochastic_green_culture  = [initial_id_green_culture[i,j][k],((final_id_green_culture[i,j][k] -  initial_id_green_culture[i,j][k])/initial_id_green_culture[i,j][k])*100]

                person_row_no_green_no_culture.append(attribute_difference_stochastic_no_green_no_culture)
                person_row_green_no_culture.append(attribute_difference_stochastic_green_no_culture)
                person_row_no_green_culture.append(attribute_difference_stochastic_no_green_culture)
                person_row_green_culture.append(attribute_difference_stochastic_green_culture)

            experiment_row_no_green_no_culture.append(person_row_no_green_no_culture)
            experiment_row_green_no_culture.append(person_row_green_no_culture)
            experiment_row_no_green_culture.append(person_row_no_green_culture)
            experiment_row_green_culture.append(person_row_green_culture)
        attribute_difference_lists_no_green_no_culture.append(experiment_row_no_green_no_culture)
        attribute_difference_lists_green_no_culture.append(experiment_row_green_no_culture)
        attribute_difference_lists_no_green_culture.append(experiment_row_no_green_culture)
        attribute_difference_lists_green_culture.append(experiment_row_green_culture)

    return np.asarray(attribute_difference_lists_no_green_no_culture),np.asarray(attribute_difference_lists_green_no_culture),np.asarray(attribute_difference_lists_no_green_culture),np.asarray(attribute_difference_lists_green_culture)


def main(RUN = 1, fileName = "results/test",BASE_PARAMS_LOAD = "package/constants/base_params_add_greens.json",dpi_save = 1200, reps = 100, green_N = 20):    
    if RUN:
        f = open(BASE_PARAMS_LOAD)
        base_params = json.load(f)
        
        ###############################################################
        
        mean_list = np.linspace(0.01,0.99, reps)
        sum_a_b = 4#set the degree of polarisation? i think the more polarised the stronger the effect will be

        init_attitudes_list = gen_atttiudes_list(mean_list, sum_a_b)# GET THE LIST

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

        ####################################################################
        # NOW DO ADDING GREENS
        
        base_params_green_culture = base_params.copy()
        base_params_green_no_culture = base_params.copy()


        base_params_green_culture["green_N"] = green_N
        base_params_green_no_culture["green_N"] = green_N
        green_K = calc_new_K(base_params_green_culture["K"],base_params_green_culture["N"], green_N)
        base_params_green_culture["K"] = green_K
        base_params_green_no_culture["K"] = green_K

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

        #############################################################

        #fileName = produceName(params, params_name)
        root = "green_influencers_culture_four_alt"
        fileName = produce_name_datetime(root)
        print("fileName:", fileName)

        ##############################################################################
        #NO GREENS
        emissions_list_no_green_no_culture, emissions_id_list_no_green_no_culture, initial_individual_carbon_emissions_id_no_green_no_culture, first_attitude_id_list_no_green_no_culture, initial_first_attitude_id_list_no_green_no_culture = multi_stochstic_emissions_run_all_individual(params_list_no_green_no_culture)
        emissions_list_no_green_culture, emissions_id_list_no_green_culture, initial_individual_carbon_emissions_id_no_green_culture, first_attitude_id_list_no_green_culture, initial_first_attitude_id_list_no_green_culture = multi_stochstic_emissions_run_all_individual(params_list_no_green_culture)
        #GREENS
        emissions_list_green_no_culture, emissions_id_list_green_no_culture, initial_individual_carbon_emissions_id_green_no_culture, first_attitude_id_list_green_no_culture, initial_first_attitude_id_list_green_no_culture = multi_stochstic_emissions_run_all_individual(params_list_green_no_culture)
        emissions_list_green_culture, emissions_id_list_green_culture, initial_individual_carbon_emissions_id_green_culture, first_attitude_id_list_green_culture, initial_first_attitude_id_list_green_culture = multi_stochstic_emissions_run_all_individual(params_list_green_culture)

        #print("first_attitude_id_list:",first_attitude_id_list_no_green_no_culture[0],first_attitude_id_list_no_green_culture[0], first_attitude_id_list_green_no_culture[0],first_attitude_id_list_green_culture[0])
        #print("first_threshold_id_list:",first_threshold_id_list_no_green_no_culture[0],first_threshold_id_list_no_green_culture[0], first_threshold_id_list_green_no_culture[0],first_threshold_id_list_green_culture[0])
        emissions_difference_matrix_compare_green,emissions_difference_matrix_compare_no_green,emissions_difference_matrix_compare_culture,emissions_difference_matrix_compare_no_culture = calc_id_attribute_change(emissions_id_list_no_green_no_culture,emissions_id_list_no_green_culture,emissions_id_list_green_no_culture,emissions_id_list_green_culture,mean_list,base_params)
        first_attitude_difference_matrix_compare_green,first_attitude_difference_matrix_compare_no_green,first_attitude_difference_matrix_compare_culture,first_attitude_difference_matrix_compare_no_culture = calc_id_attribute_change(first_attitude_id_list_no_green_no_culture,first_attitude_id_list_no_green_culture,first_attitude_id_list_green_no_culture,first_attitude_id_list_green_culture,mean_list,base_params)
        

        emissions_difference_lists_no_green_no_culture, emissions_difference_lists_green_no_culture,emissions_difference_lists_no_green_culture,emissions_difference_lists_green_culture = calc_id_attribute_change_start_finish(emissions_id_list_no_green_no_culture, emissions_id_list_green_no_culture,emissions_id_list_no_green_culture,emissions_id_list_green_culture,    initial_individual_carbon_emissions_id_no_green_no_culture,initial_individual_carbon_emissions_id_green_no_culture,initial_individual_carbon_emissions_id_no_green_culture,   initial_individual_carbon_emissions_id_green_culture ,mean_list,base_params)
        #print("emissions_difference_lists_no_green_no_culture",emissions_difference_lists_no_green_no_culture.shape)
        
        first_attitude_difference_lists_no_green_no_culture, first_attitude_difference_lists_green_no_culture,first_attitude_difference_lists_no_green_culture,first_attitude_difference_lists_green_culture = calc_id_attribute_change_start_finish(first_attitude_id_list_no_green_no_culture, first_attitude_id_list_green_no_culture,first_attitude_id_list_no_green_culture,first_attitude_id_list_green_culture,  initial_first_attitude_id_list_no_green_no_culture,initial_first_attitude_id_list_green_no_culture,initial_first_attitude_id_list_no_green_culture,   initial_first_attitude_id_list_green_culture ,mean_list,base_params)
        #quit()

        #print("initial_individual_carbon_emissions_id_no_green_no_culture", initial_individual_carbon_emissions_id_no_green_no_culture.shape)

        createFolder(fileName)

        save_object(emissions_list_no_green_no_culture, fileName + "/Data", "emissions_list_no_green_no_culture")
        save_object(emissions_list_no_green_culture, fileName + "/Data", "emissions_list_no_green_culture")
        save_object(emissions_list_green_no_culture, fileName + "/Data", "emissions_list_green_no_culture")
        save_object(emissions_list_green_culture, fileName + "/Data", "emissions_list_green_culture")

        save_object(emissions_id_list_no_green_no_culture, fileName + "/Data", "emissions_id_list_no_green_no_culture")
        save_object(emissions_id_list_no_green_culture, fileName + "/Data", "emissions_id_list_no_green_culture")
        save_object(emissions_id_list_green_no_culture, fileName + "/Data", "emissions_id_list_green_no_culture")
        save_object(emissions_id_list_green_culture, fileName + "/Data", "emissions_id_list_green_culture")

        save_object(emissions_difference_matrix_compare_green,fileName + "/Data", "emissions_difference_matrix_compare_green")
        save_object(emissions_difference_matrix_compare_no_green,fileName + "/Data", "emissions_difference_matrix_compare_no_green")
        save_object(emissions_difference_matrix_compare_culture,fileName + "/Data", "emissions_difference_matrix_compare_culture")
        save_object(emissions_difference_matrix_compare_no_culture,fileName + "/Data", "emissions_difference_matrix_compare_no_culture")

        save_object(first_attitude_difference_matrix_compare_green,fileName + "/Data", "first_attitude_difference_matrix_compare_green")
        save_object(first_attitude_difference_matrix_compare_no_green,fileName + "/Data", "first_attitude_difference_matrix_compare_no_green")
        save_object(first_attitude_difference_matrix_compare_culture,fileName + "/Data", "first_attitude_difference_matrix_compare_culture")
        save_object(first_attitude_difference_matrix_compare_no_culture,fileName + "/Data", "first_attitude_difference_matrix_compare_no_culture")

        save_object(initial_individual_carbon_emissions_id_no_green_no_culture, fileName + "/Data", "initial_individual_carbon_emissions_id_no_green_no_culture")
        save_object(initial_individual_carbon_emissions_id_no_green_culture, fileName + "/Data", "initial_individual_carbon_emissions_id_no_green_culture")
        save_object(initial_individual_carbon_emissions_id_green_no_culture, fileName + "/Data", "initial_individual_carbon_emissions_id_green_no_culture")
        save_object(initial_individual_carbon_emissions_id_green_culture, fileName + "/Data", "initial_individual_carbon_emissions_id_green_culture")

        save_object(initial_first_attitude_id_list_no_green_no_culture, fileName + "/Data", "initial_first_attitude_id_list_no_green_no_culture")
        save_object(initial_first_attitude_id_list_no_green_culture, fileName + "/Data", "initial_first_attitude_id_list_no_green_culture")
        save_object(initial_first_attitude_id_list_green_no_culture, fileName + "/Data", "initial_first_attitude_id_list_green_no_culture")
        save_object(initial_first_attitude_id_list_green_culture, fileName + "/Data", "initial_first_attitude_id_list_green_culture")

                
        save_object(emissions_difference_lists_no_green_no_culture, fileName + "/Data", "emissions_difference_lists_no_green_no_culture")
        save_object(emissions_difference_lists_no_green_culture, fileName + "/Data", "emissions_difference_lists_no_green_culture")
        save_object(emissions_difference_lists_green_no_culture, fileName + "/Data", "emissions_difference_lists_green_no_culture")
        save_object(emissions_difference_lists_green_culture, fileName + "/Data", "emissions_difference_lists_green_culture")

        save_object(first_attitude_difference_lists_no_green_no_culture, fileName + "/Data", "first_attitude_difference_lists_no_green_no_culture")
        save_object(first_attitude_difference_lists_no_green_culture, fileName + "/Data", "first_attitude_difference_lists_no_green_culture")
        save_object(first_attitude_difference_lists_green_no_culture, fileName + "/Data", "first_attitude_difference_lists_green_no_culture")
        save_object(first_attitude_difference_lists_green_culture, fileName + "/Data", "first_attitude_difference_lists_green_culture")

        save_object(mean_list, fileName + "/Data", "mean_list")
        save_object(sum_a_b , fileName + "/Data", "sum_a_b")
        save_object(base_params, fileName + "/Data", "base_params")
        save_object(init_attitudes_list,fileName + "/Data", "init_attitudes_list")
        save_object(green_N, fileName + "/Data", "green_N")
        save_object(green_K, fileName + "/Data", "green_K")            
    else:
        emissions_list_no_green_no_culture = load_object( fileName + "/Data", "emissions_list_no_green_no_culture")
        emissions_list_no_green_culture = load_object( fileName + "/Data", "emissions_list_no_green_culture")
        emissions_list_green_no_culture = load_object( fileName + "/Data", "emissions_list_green_no_culture")
        emissions_list_green_culture = load_object( fileName + "/Data", "emissions_list_green_culture")

        emissions_id_list_no_green_no_culture = load_object( fileName + "/Data", "emissions_id_list_no_green_no_culture")
        emissions_id_list_no_green_culture = load_object( fileName + "/Data", "emissions_id_list_no_green_culture")
        emissions_id_list_green_no_culture = load_object( fileName + "/Data", "emissions_id_list_green_no_culture")
        emissions_id_list_green_culture = load_object( fileName + "/Data", "emissions_id_list_green_culture")

        initial_individual_carbon_emissions_id_no_green_no_culture = load_object( fileName + "/Data", "initial_individual_carbon_emissions_id_no_green_no_culture")
        initial_individual_carbon_emissions_id_no_green_culture = load_object( fileName + "/Data", "initial_individual_carbon_emissions_id_no_green_culture")
        initial_individual_carbon_emissions_id_green_no_culture = load_object( fileName + "/Data", "initial_individual_carbon_emissions_id_green_no_culture")
        initial_individual_carbon_emissions_id_green_culture = load_object( fileName + "/Data", "initial_individual_carbon_emissions_id_green_culture")

        emissions_difference_matrix_compare_green = load_object(fileName + "/Data", "emissions_difference_matrix_compare_green")
        emissions_difference_matrix_compare_no_green = load_object(fileName + "/Data", "emissions_difference_matrix_compare_no_green")
        emissions_difference_matrix_compare_culture = load_object(fileName + "/Data", "emissions_difference_matrix_compare_culture")
        emissions_difference_matrix_compare_no_culture = load_object(fileName + "/Data", "emissions_difference_matrix_compare_no_culture")

        first_attitude_difference_matrix_compare_green = load_object(fileName + "/Data", "first_attitude_difference_matrix_compare_green")
        first_attitude_difference_matrix_compare_no_green = load_object(fileName + "/Data", "first_attitude_difference_matrix_compare_no_green")
        first_attitude_difference_matrix_compare_culture = load_object(fileName + "/Data", "first_attitude_difference_matrix_compare_culture")
        first_attitude_difference_matrix_compare_no_culture = load_object(fileName + "/Data", "first_attitude_difference_matrix_compare_no_culture")

        initial_first_attitude_id_list_no_green_no_culture = load_object( fileName + "/Data", "initial_first_attitude_id_list_no_green_no_culture")
        initial_first_attitude_id_list_no_green_culture = load_object( fileName + "/Data", "initial_first_attitude_id_list_no_green_culture")
        initial_first_attitude_id_list_green_no_culture = load_object( fileName + "/Data", "initial_first_attitude_id_list_green_no_culture")
        initial_first_attitude_id_list_green_culture = load_object( fileName + "/Data", "initial_first_attitude_id_list_green_culture")

        emissions_difference_lists_no_green_no_culture = load_object( fileName + "/Data", "emissions_difference_lists_no_green_no_culture")
        emissions_difference_lists_no_green_culture = load_object( fileName + "/Data", "emissions_difference_lists_no_green_culture")
        emissions_difference_lists_green_no_culture = load_object( fileName + "/Data", "emissions_difference_lists_green_no_culture")
        emissions_difference_lists_green_culture = load_object( fileName + "/Data", "emissions_difference_lists_green_culture")

        first_attitude_difference_lists_no_green_no_culture = load_object( fileName + "/Data", "first_attitude_difference_lists_no_green_no_culture")
        first_attitude_difference_lists_no_green_culture = load_object( fileName + "/Data", "first_attitude_difference_lists_no_green_culture")
        first_attitude_difference_lists_green_no_culture = load_object( fileName + "/Data", "first_attitude_difference_lists_green_no_culture")
        first_attitude_difference_lists_green_culture = load_object( fileName + "/Data", "first_attitude_difference_lists_green_culture")

        mean_list = load_object( fileName + "/Data", "mean_list")
        sum_a_b  = load_object( fileName + "/Data", "sum_a_b")
        base_params = load_object( fileName + "/Data", "base_params")
        init_attitudes_list = load_object(fileName + "/Data", "init_attitudes_list")
        green_N = load_object( fileName + "/Data", "green_N")
        green_K = load_object( fileName + "/Data", "green_K")


    #plot_four_compare(fileName, emissions_difference_matrix_compare_green, emissions_difference_matrix_compare_no_green, emissions_difference_matrix_compare_culture, emissions_difference_matrix_compare_no_culture, base_params["confirmation_bias"],mean_list, dpi_save)
    #plot_four(fileName, np.asarray(emissions_list_no_green_no_culture), np.asarray(emissions_list_no_green_culture), np.asarray(emissions_list_green_no_culture), np.asarray(emissions_list_green_culture), base_params["confirmation_bias"],mean_list, dpi_save)
    #plot_four_compare_one(fileName, emissions_difference_matrix_compare_green, emissions_difference_matrix_compare_no_green, emissions_difference_matrix_compare_culture, emissions_difference_matrix_compare_no_culture, mean_list, dpi_save)
    #plot_four_two(fileName, emissions_difference_matrix_compare_green, emissions_difference_matrix_compare_no_green, emissions_difference_matrix_compare_culture, emissions_difference_matrix_compare_no_culture, mean_list, dpi_save)
    #plot_four_two_attitude(fileName, emissions_difference_matrix_compare_green, emissions_difference_matrix_compare_no_green, emissions_difference_matrix_compare_culture, emissions_difference_matrix_compare_no_culture, mean_list, dpi_save)

    #Emissions
    #plot_id_change(fileName,emissions_difference_lists_no_green_no_culture, initial_individual_carbon_emissions_id_no_green_no_culture,  mean_list, dpi_save, r"Emissions","No green, No culture")
    #plot_id_change(fileName,emissions_difference_lists_green_no_culture, initial_individual_carbon_emissions_id_green_no_culture,  mean_list, dpi_save, r"Emissions","Green, No culture")
    #plot_id_change(fileName,emissions_difference_lists_no_green_culture, initial_individual_carbon_emissions_id_no_green_culture,  mean_list, dpi_save, r"Emissions","No green, Culture")
    #plot_id_change(fileName,emissions_difference_lists_green_culture, initial_individual_carbon_emissions_id_green_culture,  mean_list, dpi_save, r"Emissions","Green, Culture")

    #First attitude
    #plot_id_change(fileName,first_attitude_difference_lists_no_green_no_culture, initial_first_attitude_id_list_no_green_no_culture,  mean_list, dpi_save, r"Attitude ($m = 1$)","No green, No culture")
    #plot_id_change(fileName,first_attitude_difference_lists_green_no_culture, initial_first_attitude_id_list_green_no_culture,  mean_list, dpi_save, r"Attitude ($m = 1$)","Green, No culture")
    #plot_id_change(fileName,first_attitude_difference_lists_no_green_culture, initial_first_attitude_id_list_no_green_culture,  mean_list, dpi_save, r"Attitude ($m = 1$)","No green, Culture")
    #plot_id_change(fileName,first_attitude_difference_lists_green_culture, initial_first_attitude_id_list_green_culture,  mean_list, dpi_save,r"Attitude ($m = 1$)","Green, Culture")

    #plot_id_change(fileName,emissions_difference_lists_no_green_no_culture, emissions_difference_lists_green_no_culture,emissions_difference_lists_no_green_culture,emissions_difference_lists_green_culture,       initial_individual_carbon_emissions_id_no_green_no_culture, initial_individual_carbon_emissions_id_green_no_culture, initial_individual_carbon_emissions_id_no_green_culture, initial_individual_carbon_emissions_id_green_culture,  mean_list, dpi_save, r"Emissions")
    #plot_id_change(fileName,first_attitude_difference_lists_no_green_no_culture, first_attitude_difference_lists_no_green_culture,first_attitude_difference_lists_green_no_culture,first_attitude_difference_lists_green_culture,      initial_first_attitude_id_list_no_green_no_culture, initial_first_attitude_id_list_green_no_culture, initial_first_attitude_id_list_no_green_culture, initial_first_attitude_id_list_green_culture, mean_list, dpi_save, r"Attitude ($m = 1$)")
    cases_list_emissions = np.asarray([emissions_difference_lists_no_green_no_culture,emissions_difference_lists_green_no_culture,emissions_difference_lists_no_green_culture,emissions_difference_lists_green_culture])
    cases_list_attitude = np.asarray([first_attitude_difference_lists_no_green_no_culture,first_attitude_difference_lists_green_no_culture,first_attitude_difference_lists_no_green_culture,first_attitude_difference_lists_green_culture])
    
    case_name_list = ["No green, No culture","Green , No culture", "No green, Culture", "Green, Culture"]
    av_reps = len(base_params["seed_list"])

    plot_id_change_cases(fileName, cases_list_emissions , mean_list, dpi_save, r"Emissions",case_name_list,av_reps)
    plot_id_change_cases(fileName, cases_list_attitude , mean_list, dpi_save, r"Attitude ($m = 1$)",case_name_list,av_reps)

    plt.show()

    return fileName

if __name__ == '__main__':
    main(RUN = 1, fileName = "results/test",BASE_PARAMS_LOAD = "package/constants/base_params_alt.json",dpi_save = 1200, reps = 100,green_N = 20)