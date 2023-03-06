"""Plot data from runs including effect of green influencers

Created: 10/10/2022
"""
import matplotlib.pyplot as plt
from package.resources.utility import (
    load_object
)
from package.resources.plot import (
    plot_diff_emissions_comparison,
    plot_single
)

def load_data(fileName):
    data_dict = {}
    data_dict["carbon_emissions_not_influencer_no_green_no_identity"] = load_object( fileName + "/Data", "carbon_emissions_not_influencer_no_green_no_culture")
    data_dict["carbon_emissions_not_influencer_no_green_identity"] = load_object( fileName + "/Data", "carbon_emissions_not_influencer_no_green_culture")
    data_dict["carbon_emissions_not_influencer_green_no_identity"] = load_object( fileName + "/Data", "carbon_emissions_not_influencer_green_no_culture")
    data_dict["carbon_emissions_not_influencer_green_identity"] = load_object( fileName + "/Data", "carbon_emissions_not_influencer_green_culture")

    data_dict["emissions_difference_matrix_compare_green"] = load_object(fileName + "/Data", "emissions_difference_matrix_compare_green")
    data_dict["emissions_difference_matrix_compare_no_green"] = load_object(fileName + "/Data", "emissions_difference_matrix_compare_no_green")
    data_dict["emissions_difference_matrix_compare_identity"] = load_object(fileName + "/Data", "emissions_difference_matrix_compare_culture")
    data_dict["emissions_difference_matrix_compare_no_identity"] = load_object(fileName + "/Data", "emissions_difference_matrix_compare_no_culture")

    data_dict["mean_list"] = load_object( fileName + "/Data", "mean_list")
    data_dict["sum_a_b"]  = load_object( fileName + "/Data", "sum_a_b")
    data_dict["base_params"] = load_object( fileName + "/Data", "base_params")
    data_dict["init_attitudes_list"] = load_object(fileName + "/Data", "init_attitudes_list")
    data_dict["green_N"]= load_object( fileName + "/Data", "green_N")
    return data_dict

def main(fileName_list = ["results/green_influencers_culture_four_alt_18_00_22__20_02_2023"],dpi_save = 1200, latex_bool = 0):    

    data_dict_list = []
    for fileName in fileName_list:
        data_dict_list.append(load_data(fileName))

    #plot_diff_emissions_comparison(data_dict_list, fileName_list,dpi_save, latex_bool = latex_bool)
    plot_single(data_dict_list, fileName_list,dpi_save)
    plt.show()  

if __name__ == '__main__':
    main(fileName_list = ["results/green_influencers_culture_four_alt_17_59_22__20_02_2023"],dpi_save = 1200, latex_bool = 0)
