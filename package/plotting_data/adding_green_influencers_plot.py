"""Plot data from runs including effect of green influencers
Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from package.resources.plot import (
    plot_emissions_multi_ab_min_max_two_theta_reverse_add_green,
)
from package.resources.utility import (
    load_object,
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
    fileName_add_greens_theta_one = "results/splitting_eco_warriors_single_add_greens_17_44_05__01_02_2023",
    fileName_add_greens_theta_two = "results/splitting_eco_warriors_single_add_greens_17_44_05__01_02_2023",
    dpi_save = 1200,
    latex_bool = 0
    ) -> None: 

    #emissions_list_default_theta_one = load_object( fileName_add_greens_theta_one + "/Data", "emissions_list_default")
    #emissions_list_add_green_theta_one  = load_object( fileName_add_greens_theta_one + "/Data", "emissions_list_add_green")
    #emissions_id_list_individual_default_theta_one = load_object( fileName_add_greens_theta_one + "/Data", "emissions_id_list_individual_default")
    #emissions_id_list_individual_add_green_theta_one  = load_object( fileName_add_greens_theta_one + "/Data", "emissions_id_list_individual_add_green")
    base_params_theta_one = load_object( fileName_add_greens_theta_one + "/Data", "base_params")
    #init_attitudes_list_theta_one = load_object(fileName_add_greens_theta_one + "/Data", "init_attitudes_list")
    mean_list_theta_one = load_object(fileName_add_greens_theta_one + "/Data", "mean_list")
    #sum_a_b_theta_one = load_object(fileName_add_greens_theta_one + "/Data", "sum_a_b")
    #green_N_theta_one = load_object( fileName_add_greens_theta_one + "/Data", "green_N")
    #green_K_theta_one = load_object(fileName_add_greens_theta_one + "/Data", "green_K")
    emissions_difference_matrix_theta_one = load_object(fileName_add_greens_theta_one + "/Data", "emissions_difference_matrix")

    #emissions_list_default_theta_two = load_object( fileName_add_greens_theta_two + "/Data", "emissions_list_default")
    #emissions_list_add_green_theta_two  = load_object( fileName_add_greens_theta_two + "/Data", "emissions_list_add_green")
    #emissions_id_list_individual_default_theta_two = load_object( fileName_add_greens_theta_two + "/Data", "emissions_id_list_individual_default")
    #emissions_id_list_individual_add_green_theta_two  = load_object( fileName_add_greens_theta_two + "/Data", "emissions_id_list_individual_add_green")
    base_params_theta_two = load_object( fileName_add_greens_theta_two + "/Data", "base_params")
    #init_attitudes_list_theta_two = load_object(fileName_add_greens_theta_two + "/Data", "init_attitudes_list")
    #mean_list_theta_two = load_object(fileName_add_greens_theta_two + "/Data", "mean_list")
    #sum_a_b_theta_two = load_object(fileName_add_greens_theta_two + "/Data", "sum_a_b")
    #green_N_theta_two = load_object( fileName_add_greens_theta_two + "/Data", "green_N")
    #green_K_theta_two = load_object(fileName_add_greens_theta_two + "/Data", "green_K")
    emissions_difference_matrix_theta_two = load_object(fileName_add_greens_theta_two + "/Data", "emissions_difference_matrix")

    plot_emissions_multi_ab_min_max_two_theta_reverse_add_green(fileName_add_greens_theta_one,   emissions_difference_matrix_theta_one,    emissions_difference_matrix_theta_two, base_params_theta_one["confirmation_bias"],base_params_theta_two["confirmation_bias"],   mean_list_theta_one,   dpi_save, len(base_params_theta_one["seed_list"]),latex_bool = latex_bool )
    
    plt.show()
