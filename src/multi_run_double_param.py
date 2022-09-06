from logging import raiseExceptions
from run import parallel_run
import matplotlib.pyplot as plt
import numpy as np
from network import Network
from utility import createFolderSA,produce_param_list_double,generate_title_list
from matplotlib.colors import LinearSegmentedColormap,  Normalize
from matplotlib.cm import get_cmap
from plot import (
    live_print_culture_timeseries,
    print_culture_timeseries_vary_array,
    live_print_culture_timeseries_vary,
    live_compare_plot_animate_behaviour_scatter,
    live_phase_diagram_k_means_vary,
    print_culture_time_series_clusters_two_properties,
    live_compare_animate_culture_network_and_weighting,
    live_compare_animate_weighting_matrix,
    live_compare_animate_behaviour_matrix,
    live_print_heterogenous_culture_momentum_double,
)

params = {
    "total_time": 200,
    "delta_t": 0.05,
    "compression_factor": 10,
    "save_data": True, 
    "alpha_change" : 1.0,
    "harsh_data": False,
    "averaging_method": "Arithmetic",
    "phi_list_lower": 0.1,
    "phi_list_upper": 1.0,
    "N": 200,
    "M": 3,
    "K": 20,
    "prob_rewire": 0.05,
    "set_seed": 1,
    "culture_momentum_real": 5,
    "learning_error_scale": 0.02,
    "discount_factor": 0.6,
    "present_discount_factor": 0.8,
    "inverse_homophily": 0.1,#1 is total mixing, 0 is no mixing
    "homophilly_rate" : 1.5,
    "confirmation_bias": 30,
}

params["time_steps_max"] = int(params["total_time"] / params["delta_t"])

#behaviours!
if params["harsh_data"]:#trying to create a polarised society!
    params["green_extreme_max"]= 8
    params["green_extreme_min"]= 2
    params["green_extreme_prop"]= 2/5
    params["indifferent_max"]= 2
    params["indifferent_min"]= 2
    params["indifferent_prop"]= 1/5
    params["brown_extreme_min"]= 2
    params["brown_extreme_max"]= 8
    params["brown_extreme_prop"]= 2/5
    if params["green_extreme_prop"] + params["indifferent_prop"] + params["brown_extreme_prop"] != 1:
        raise Exception("Invalid proportions")
else:
    params["alpha_attitude"] = 1
    params["beta_attitude"] = 1
    params["alpha_threshold"] = 1
    params["beta_threshold"] = 1

###PLOT STUFF
node_size = 50
cmap = LinearSegmentedColormap.from_list("BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1)

#norm_neg_pos = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=-1, vmax=1))
norm_neg_pos = Normalize(vmin=-1, vmax=1)
norm_zero_one  = Normalize(vmin=0, vmax=1)

#log_norm = SymLogNorm(linthresh=0.15, linscale=1, vmin=-1.0, vmax=1.0, base=10)  # this works at least its correct
cmap_weighting = "Reds"
cmap_edge = get_cmap("Greys")
fps = 5
interval = 50
layout = "circular"
round_dec = 2

nrows = 2
ncols = 3

alpha_quick, alpha_normal, alpha_lagard = 0.9,0.7,0.9
colour_quick, colour_normal, colour_lagard = "indianred", "grey", "cornflowerblue"

#print("time_steps_max", time_steps_max)
frame_num = ncols * nrows - 1

min_val = 1e-3

dpi_save = 1200

min_k,max_k = 2,10#N - 1# Cover all the possible bases with the max k though it does slow it down
alpha_val = 0.25
size_points = 5
min_culture_distance = 0.5

if __name__ == "__main__":

        nrows = 4
        ncols = 4#due to screen ratio want more cols than rows usually
        reps = nrows*ncols

        param_min_row = 0.1
        param_max_row = 5
        param_min_col = 0.1
        param_max_col = 5#50.0

        property_row = r"$\alpha$"#"Heteogenity proportion" #"Confirmation bias"
        param_row = "alpha_attitude"#"quick_changers_prop"
        property_col = r"$\beta$" #"Cultural momentum"#"Inverse homophily"
        param_col = "beta_attitude"#"culture_momentum_real"#"inverse_homophily"


        fileName = "results/%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (param_col,param_row,str(params["N"]),str(params["time_steps_max"]),str(params["K"]),str(param_min_col), str(param_max_col), str(param_min_row), str(param_max_row), str(reps))
        print("fileName: ", fileName)
        createFolderSA(fileName)

        ### GENERATE DATA
        #print(np.linspace(param_min_row,param_max_row, nrows), type(np.linspace(param_min_row,param_max_row, nrows)))
        #print(np.asarray([0.05, 0.5, 1.0, 5.0]), type(np.asarray([0.05, 0.5, 1.0, 5.0])))
        #quit()
        row_list = np.asarray([0.08, 0.5, 1.0, 5.0])#np.linspace(param_min_row,param_max_row, nrows)
        col_list = np.asarray([0.08, 0.5, 1.0, 5.0])#np.linspace(param_min_col,param_max_col, ncols)
        params_list = produce_param_list_double(params,param_col,col_list,param_row,row_list)
        
        data_list  = parallel_run(params_list)  
        data_array = np.reshape(data_list, (len(row_list), len(col_list)))
        title_list = generate_title_list(property_col,col_list,property_row,row_list, round_dec)


        ### PLOTS
        live_print_culture_timeseries_vary(fileName, data_list, param_row, param_col,title_list, nrows, ncols,  dpi_save)
        #print_culture_timeseries_vary_array(fileName, data_array, property_col, col_list,property_row,row_list,  nrows, ncols , dpi_save)
        #live_phase_diagram_k_means_vary(fileName, data_array, property_row,  row_list,property_col,col_list,min_k,max_k,size_points, cmap_weighting,dpi_save)
        #print_culture_time_series_clusters_two_properties(fileName,data_array, row_list, col_list,property_row, property_col, min_k,max_k,size_points, alpha_val, min_culture_distance,"DTW", nrows, ncols, dpi_save, round_dec)

        #ani_b = live_compare_animate_culture_network_and_weighting(fileName,data_list,layout,cmap,node_size,interval,fps,norm_zero_one,round_dec,cmap_edge, ncols, nrows,property_col,col_list)
        #ani_c = live_compare_animate_weighting_matrix(fileName, data_list,  cmap_weighting, interval, fps, round_dec, cmap_edge, nrows, ncols,property_col,col_list)
        #ani_d = live_compare_animate_behaviour_matrix(fileName, data_list,  cmap, interval, fps, round_dec, nrows, ncols,property_col,col_list)

        plt.show()