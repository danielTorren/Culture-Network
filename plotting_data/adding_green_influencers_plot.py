"""Plot data from runs including effect of green influencers
Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from resources.plot import (
    plot_emissions_multi_ab_relative_all_add_green,
)
from resources.utility import (
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
    fileName = "results/splitting_eco_warriors_single_add_greens_17_44_05__01_02_2023",
    dpi_save = 1200
    ) -> None: 

    emissions_list_default = load_object( fileName + "/Data", "emissions_list_default")
    emissions_list_add_green  = load_object( fileName + "/Data", "emissions_list_add_green")
    base_params = load_object( fileName + "/Data", "base_params")
    mean_list = load_object(fileName + "/Data", "mean_list")

    plot_emissions_multi_ab_relative_all_add_green(fileName, emissions_list_default, emissions_list_add_green, mean_list, dpi_save, len(base_params["seed_list"]))
    
    plt.show()
