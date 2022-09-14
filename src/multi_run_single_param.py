from logging import raiseExceptions
from run import parallel_run,average_seed_parallel_run_mena_coefficient_variance
import matplotlib.pyplot as plt
import numpy as np
from network import Network
from utility import createFolderSA,produce_param_list
from matplotlib.colors import LinearSegmentedColormap,  Normalize,LogNorm
from matplotlib.cm import get_cmap
from plot import (
    print_culture_time_series_generic,
    plot_average_culture_comparison,
    plot_carbon_emissions_total_comparison,
    plot_weighting_matrix_convergence_comparison,
    plot_average_culture_no_range_comparison,
    plot_live_link_change_comparison,
    plot_live_cum_link_change_comparison,
    plot_live_link_change_per_agent_comparison,
    plot_live_cum_link_change_per_agent_comparison,
    print_culture_time_series_clusters,
    print_live_intial_culture_networks,
    prints_init_weighting_matrix,
    prints_final_weighting_matrix,
    multi_animation_weighting,
    live_compare_animate_culture_network_and_weighting,
    live_compare_animate_weighting_matrix,
    live_compare_animate_behaviour_matrix,
    live_compare_plot_animate_behaviour_scatter,
    live_print_heterogenous_culture_momentum,
    live_print_culture_timeseries,
    live_print_culture_timeseries_with_weighting,
    print_live_intial_culture_networks_and_culture_timeseries,
    live_multirun_diagram_mean_coefficient_variance,
    live_average_multirun_diagram_mean_coefficient_variance,
)

params = {
    "total_time": 50,
    "delta_t": 0.05,
    "compression_factor": 10,
    "save_data": True, 
    "alpha_change" : 1.0,
    "harsh_data": False,
    "averaging_method": "Arithmetic",
    "phi_list_lower": 0.1,
    "phi_list_upper": 1.0,
    "N": 100,
    "M": 3,
    "K": 20,
    "prob_rewire": 0.05,
    "set_seed": 1,
    "culture_momentum_real": 5,
    "learning_error_scale": 0.02,
    "discount_factor": 0.6,
    "present_discount_factor": 0.8,
    "inverse_homophily": 0.2,#1 is total mixing, 0 is no mixing
    "homophilly_rate" : 1,
    "confirmation_bias": 20,
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
nrows_behave = 1
ncols_behave = params["M"]
node_size = 50
cmap = LinearSegmentedColormap.from_list("BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1)

#norm_neg_pos = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=-1, vmax=1))
norm_neg_pos = Normalize(vmin=-1, vmax=1)
norm_zero_one  = Normalize(vmin=0, vmax=1)

cmap_weighting = get_cmap("Reds")
cmap_edge = get_cmap("Greys")
fps = 5
interval = 50
layout = "circular"
round_dec = 2

alpha_quick, alpha_normal, alpha_lagard = 0.9,0.7,0.9
colour_quick, colour_normal, colour_lagard = "indianred", "grey", "cornflowerblue"


min_val = 1e-3
dpi_save = 600#1200

min_k,max_k = 2,10#N - 1# Cover all the possible bases with the max k though it does slow it down
alpha_val = 0.25
size_points = 5
min_culture_distance = 0.5

SINGLE = 0

if __name__ == "__main__":

    nrows = 1
    ncols = 24#48#48#due to screen ratio want more cols than rows usually
    reps = nrows*ncols# make multiples of the number of cores for efficieny

    property_varied = "confirmation_bias"#"alpha_change"#"culture_momentum"#"confirmation_bias"#"inverse_homophily" #MAKE SURE ITS TYPES CORRECTLY
    property_varied_title = "Confirmation bias"
    param_min = -1.0
    param_max = 2.0#50.0
    #title_list = [r"Static uniform $\alpha_{n,k}$", r"Static culturally determined $\alpha_{n,k}$", r"Dynamic culturally determined $\alpha_{n,k}$"]

    #property_values_list = np.linspace(param_min,param_max, reps) #np.asarray([0.0, 0.5, 1.0])#np.linspace(param_min,param_max, reps)
    property_values_list = np.logspace(param_min,param_max, reps)
    #data = parallel_run(params_list)#better if a Multiple of 4
    log_norm = LogNorm()#cant take log of negative numbers, unlike log s

    print(property_values_list)
    #property_values_list = SymLogNorm(linthresh=0.15, linscale=1, vmin=param_min, vmax=1.0, base=10)  # this works at least its correct

    if SINGLE:
    
        fileName = "results/%s_variation_%s_%s_%s_%s_%s_%s" % (property_varied,str(params["N"]),str(params["time_steps_max"]),str(params["K"]), str(param_min), str(param_max), str(reps))
        print("fileName: ", fileName)
        createFolderSA(fileName)

        params_list = produce_param_list(params,property_values_list, property_varied)
        data = parallel_run(params_list)#better if a Multiple of 4

        ###WORKING 

        #print_culture_time_series_generic(fileName, data, property_values_list, property_varied_title, dpi_save,nrows, ncols,round_dec)
        #live_print_culture_timeseries(fileName, data, property_varied, title_list, nrows, ncols, dpi_save)
        #plot_average_culture_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        #plot_carbon_emissions_total_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        #plot_weighting_matrix_convergence_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        #plot_average_culture_no_range_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        #plot_live_link_change_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        #plot_live_cum_link_change_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        #plot_live_link_change_per_agent_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        #plot_live_cum_link_change_per_agent_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        live_multirun_diagram_mean_coefficient_variance(fileName, data, property_varied, property_values_list,property_varied_title, cmap,dpi_save,norm_zero_one)
    

        #print_culture_time_series_clusters(fileName, data, property_values_list, property_varied_title, min_k,max_k,size_points, alpha_val, min_culture_distance, nrows, ncols, dpi_save, round_dec)
        #print_live_intial_culture_networks(fileName, data, dpi_save, property_values_list, property_varied, nrows, ncols , layout, norm_zero_one, cmap, node_size,round_dec)
        #prints_init_weighting_matrix(fileName, data, dpi_save,nrows, ncols, cmap_weighting,property_values_list, property_varied,round_dec)
        #prints_final_weighting_matrix(fileName, data, dpi_save,nrows, ncols, cmap_weighting,property_values_list, property_varied,round_dec)
        #live_print_culture_timeseries_with_weighting(fileName, data, property_varied, title_list, nrows, ncols, dpi_save, cmap_weighting)
        #print_live_intial_culture_networks_and_culture_timeseries(fileName, data, dpi_save, property_values_list, property_varied_title, ncols, layout, norm_zero_one, cmap, node_size,round_dec)

        #ani_a =  multi_animation_weighting(fileName,data, cmap_weighting,  interval, fps, round_dec, nrows, ncols)
        #ani_b = live_compare_animate_culture_network_and_weighting(fileName,data,layout,cmap,node_size,interval,fps,norm_zero_one,round_dec,cmap_edge, ncols, nrows,property_varied_title,property_values_list)
        #ani_c = live_compare_animate_weighting_matrix(fileName, data,  cmap_weighting, interval, fps, round_dec, cmap_edge, nrows, ncols,property_varied_title,property_values_list)
        #ani_d = live_compare_animate_behaviour_matrix(fileName, data,  cmap, interval, fps, round_dec, nrows, ncols,property_varied_title,property_values_list)
        #ani_e = live_compare_plot_animate_behaviour_scatter(fileName,data,norm_zero_one, cmap, nrows, ncols,property_varied, property_values_list,interval, fps,round_dec)
    else:
        #AVERAGE OVER MULTIPLE RUNS
        seed_list = [1,2,3,4,5]#ie 5 reps per run!
        params["seed_list"] = seed_list
        average_reps = len(seed_list)

        fileName = "results/average_%s_variation_%s_%s_%s_%s_%s_%s_%s" % (property_varied,str(params["N"]),str(params["time_steps_max"]),str(params["K"]), str(param_min), str(param_max), str(reps), str(average_reps))
        print("fileName: ", fileName)
        createFolderSA(fileName)

        params_list = produce_param_list(params,property_values_list, property_varied)
        mean_data, coefficient_variance_data = average_seed_parallel_run_mena_coefficient_variance(params_list)




        
        live_average_multirun_diagram_mean_coefficient_variance(fileName, mean_data,coefficient_variance_data, property_varied, property_values_list,property_varied_title, cmap_weighting, dpi_save, log_norm)

        plt.show()



