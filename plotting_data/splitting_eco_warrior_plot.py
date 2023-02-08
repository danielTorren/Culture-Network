"""Plot data from runs including effect of green influencers
Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from resources.plot import (
    double_matrix_plot,
    multi_line_matrix_plot_difference,
    multi_line_matrix_plot_difference_percentage,
    plot_culture_time_series_emissions,
    plot_behaviours_time_series_emissions,
    plot_behaviours_time_series_emissions_and_culture,
    plot_emissions_distance,
    plot_emissions_multi_ab,
    plot_emissions_multi_ab_relative,
    plot_emissions_multi_ab_relative_all,
    plot_emissions_multi_ab_all,
    plot_behaviours_time_series_culture_and_emissions_ab_relative_all,
    plot_compare_emissions_adding_green,
    live_print_culture_timeseries_green,
    plot_emissions_multi_ab_relative_all_add_green,
    bifurcation_plot_add_green,
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
    PLOT_NAME = "SINGLE",
    fileName_DISTANCE_SINGLE_TIME_SERIES = "results/splitting_eco_warriors_distance_single_10_52_16__01_02_2023",
    fileName_MULTI_THETA_M_no_identity = "test",
    dpi_save = 1200
    ) -> None: 

    

    if PLOT_NAME == "MULTI_THETA_M":
            base_params = load_object(fileName + "/Data", "base_params")
            print("alpha state: ", base_params["alpha_change"])
            variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
            matrix_emissions = load_object(fileName + "/Data", "matrix_emissions")
    elif PLOT_NAME == "SINGLE_TIME_SERIES":
        Data_culture = load_object( fileName + "/Data", "Data_culture")
        Data_no_culture = load_object( fileName + "/Data", "Data_no_culture")
        base_params = load_object( fileName + "/Data", "base_params")
    elif PLOT_NAME == "DISTANCE_SINGLE_TIME_SERIES":
        data_list_culture = load_object( fileName + "/Data", "data_list_culture")
        data_list_no_culture  = load_object( fileName + "/Data", "data_list_no_culture")
        base_params = load_object( fileName + "/Data", "base_params")
        init_attitudes_list = load_object(fileName + "/Data", "init_attitudes_list")
    elif PLOT_NAME == "MULTI_A_B":
        emissions_list_culture = load_object( fileName + "/Data", "emissions_list_culture")
        emissions_list_no_culture  = load_object( fileName + "/Data", "emissions_list_no_culture")
        base_params = load_object( fileName + "/Data", "base_params")
        init_attitudes_list = load_object(fileName + "/Data", "init_attitudes_list")
        mean_list = load_object(fileName + "/Data", "mean_list")
        sum_a_b = load_object(fileName + "/Data", "sum_a_b ")
    elif PLOT_NAME == "MULTI_A_B_and_DISTANCE_SINGLE_TIME_SERIES":
        data_list_culture = load_object( fileName_DISTANCE_SINGLE_TIME_SERIES + "/Data", "data_list_culture")
        data_list_no_culture  = load_object( fileName_DISTANCE_SINGLE_TIME_SERIES + "/Data", "data_list_no_culture")
        emissions_list_culture = load_object( fileName + "/Data", "emissions_list_culture")
        emissions_list_no_culture  = load_object( fileName + "/Data", "emissions_list_no_culture")
        base_params = load_object( fileName + "/Data", "base_params")
        init_attitudes_list = load_object(fileName + "/Data", "init_attitudes_list")
        mean_list = load_object(fileName + "/Data", "mean_list")
        sum_a_b = load_object(fileName + "/Data", "sum_a_b ")
    elif PLOT_NAME == "ADD_GREENS_SINGLE":
        Data_no_greens = load_object( fileName + "/Data", "Data_no_greens")
        Data_add_greens = load_object( fileName + "/Data", "Data_add_greens")
        base_params = load_object( fileName + "/Data", "base_params")
        base_params_add_green = load_object( fileName + "/Data", "base_params_add_green")
    elif PLOT_NAME == "ADD_GREENS_MULTI_A_B":
        emissions_list_default = load_object( fileName + "/Data", "emissions_list_default")
        emissions_list_add_green  = load_object( fileName + "/Data", "emissions_list_add_green")
        emissions_id_list_individual_default = load_object( fileName + "/Data", "emissions_id_list_individual_default")
        emissions_id_list_individual_add_green  = load_object( fileName + "/Data", "emissions_id_list_individual_add_green")
        base_params = load_object( fileName + "/Data", "base_params")
        init_attitudes_list = load_object(fileName + "/Data", "init_attitudes_list")
        mean_list = load_object(fileName + "/Data", "mean_list")
        sum_a_b = load_object(fileName + "/Data", "sum_a_b")
        green_N = load_object( fileName + "/Data", "green_N")
        green_K = load_object(fileName + "/Data", "green_K")
        emissions_difference_matrix = load_object(fileName + "/Data", "emissions_difference_matrix")

    if PLOT_NAME == "MULTI_THETA_M":
        #### two D plot of emissions with confimation bias and number of eco warriors
        col_dict = variable_parameters_dict["col"]
        row_dict = variable_parameters_dict["row"]

        #### FOR confimation bias vs attitude polarisation
        max_x_val = col_dict["max"]
        min_x_val = col_dict["min"]
        col_ticks_label = list(range(min_x_val, max_x_val, 10))#[-10,0,10,20,30,40,50,60,70,80,90]#[-10,0,10,20,30,40,50,60]#[col_dict["vals"][x] for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
        col_ticks_pos = list(range(min_x_val, max_x_val, 10))#[-10,0,10,20,30,40,50,60,70,80,90]#[int(round(index_len_x_matrix*((x - min_x_val)/(max_x_val- min_x_val)))) for x in col_ticks_label]#[0,30,70,50]#[0,10,20,30,40,50,60,70]#[x for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
        print("out col_ticks_pos",col_ticks_pos)

        max_y_val = row_dict["max"]
        min_y_val = row_dict["min"]
        row_ticks_label = list(range(min_y_val, max_y_val, 10))#[0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]#[-10,0,10,20,30,40,50,60]#[col_dict["vals"][x] for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
        row_ticks_pos = list(range(min_y_val, max_y_val, 10))#[0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]#[int(round(index_len_y_matrix*((y - min_y_val)/(max_y_val- min_y_val)))) for y in col_ticks_label]#[0,30,70,50]#[0,10,20,30,40,50,60,70]#[x for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
        
        row_label = r"Number of behaviours per agent, M"#r"Eco-warriors count"#
        col_label = r'Confirmation bias, $\theta$'#r'Confirmation bias, $\theta$'
        y_label = r"Change in final emissions, $\Delta E$"#r"Identity variance, $\sigma^2$"

        matrix_emissions_no_identity = load_object(fileName_MULTI_THETA_M_no_identity + "/Data", "matrix_emissions")
        
        difference_emissions_matrix = matrix_emissions - matrix_emissions_no_identity

        difference_emissions_matrix_percentage = ((matrix_emissions - matrix_emissions_no_identity)/matrix_emissions_no_identity)*100

        multi_line_matrix_plot_difference(fileName,difference_emissions_matrix, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save,col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label, 0, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label
        multi_line_matrix_plot_difference(fileName,difference_emissions_matrix, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save,col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label, 1, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label

        multi_line_matrix_plot_difference_percentage(fileName,difference_emissions_matrix_percentage, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save,col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label, 0, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label
        multi_line_matrix_plot_difference_percentage(fileName,difference_emissions_matrix_percentage, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save,col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label, 1, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label

        double_matrix_plot(fileName,difference_emissions_matrix, y_label, "emissions",variable_parameters_dict, get_cmap("plasma"),dpi_save,col_ticks_pos,row_ticks_pos,col_ticks_label,row_ticks_label)
        double_matrix_plot(fileName,difference_emissions_matrix_percentage, y_label, "emissions_percent",variable_parameters_dict, get_cmap("plasma"),dpi_save,col_ticks_pos,row_ticks_pos,col_ticks_label,row_ticks_label)
    elif PLOT_NAME == "SINGLE_TIME_SERIES":
        plot_culture_time_series_emissions(fileName,Data_culture, Data_no_culture, dpi_save)
        plot_behaviours_time_series_emissions(fileName,Data_culture, Data_no_culture, dpi_save)
        plot_behaviours_time_series_emissions_and_culture(fileName,Data_culture, Data_no_culture, dpi_save)
    elif PLOT_NAME == "DISTANCE_SINGLE_TIME_SERIES":
        plot_emissions_distance(fileName,data_list_culture, data_list_no_culture,init_attitudes_list, dpi_save)
    elif PLOT_NAME == "MULTI_A_B":
        plot_emissions_multi_ab(fileName, emissions_list_culture, emissions_list_no_culture, mean_list, dpi_save)
        plot_emissions_multi_ab_relative(fileName, emissions_list_culture, emissions_list_no_culture, mean_list, dpi_save)
        plot_emissions_multi_ab_all(fileName, emissions_list_culture, emissions_list_no_culture, mean_list, dpi_save, len(base_params["seed_list"]))
        plot_emissions_multi_ab_relative_all(fileName, emissions_list_culture, emissions_list_no_culture, mean_list, dpi_save, len(base_params["seed_list"]))
    elif PLOT_NAME == "MULTI_A_B_and_DISTANCE_SINGLE_TIME_SERIES":
        plot_behaviours_time_series_culture_and_emissions_ab_relative_all(fileName, data_list_culture, data_list_no_culture, emissions_list_culture, emissions_list_no_culture, mean_list, dpi_save, len(base_params["seed_list"]))
    elif PLOT_NAME == "ADD_GREENS_SINGLE":
        plot_compare_emissions_adding_green(fileName,Data_no_greens,Data_add_greens,dpi_save)
        live_print_culture_timeseries_green(fileName, [Data_no_greens,Data_add_greens], "Green_Influencers", ["Default", "Add Green Influencers"],1, 2, dpi_save,["#4421af","#5ad45a"])
    elif PLOT_NAME == "ADD_GREENS_MULTI_A_B":
        plot_emissions_multi_ab_relative_all_add_green(fileName, emissions_list_default, emissions_list_add_green, mean_list, dpi_save, len(base_params["seed_list"]))
        bifurcation_plot_add_green(fileName,emissions_difference_matrix, mean_list, dpi_save)
    
    plt.show()
