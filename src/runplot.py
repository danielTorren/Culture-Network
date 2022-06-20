from run import run
from plot import (
    plot_culture_timeseries,
    animate_weighting_matrix,
    animate_behavioural_matrix,
    animate_culture_network,
    prints_behavioural_matrix,
    prints_culture_network,
    multi_animation,
    multi_animation_alt,
    multi_animation_scaled,
    plot_value_timeseries,
    plot_threshold_timeseries,
    plot_attract_timeseries,
    standard_behaviour_timeseries_plot,
    plot_carbon_price_timeseries,
    plot_total_carbon_emissions_timeseries,
    plot_av_carbon_emissions_timeseries,
    prints_weighting_matrix,
    plot_weighting_matrix_convergence_timeseries,
    plot_cultural_range_timeseries,
    plot_average_culture_timeseries,
    plot_beta_distributions,
)
from utility import loadData, get_run_properties, frame_distribution_prints
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
import time
import numpy as np

# Params
# fixed_params = [opinion_dynamics,save_data, time_steps_max, delta_t]
#[("phi_list_lower",0,0.7),("phi_list_upper",0.7,1),("N",50,500),("M",1,10),("K",2,50),("prob_rewire",0.001,0.5), ("set_seed",0,10000), ("culture_momentum",0,20),("learning_error_scale",0,0.5),("alpha_attract",0.1,10),("beta_attract",0.1,10),("alpha_threshold",0.1,10),("beta_threshold",0.1,10)]


save_data = True
opinion_dynamics = "DEGROOT"  # "SELECT"
K = 80  # k nearest neighbours INTEGER
M = 3  # number of behaviours
N = 500  # number of agents
total_time = 100

alpha_attract = 1  ##inital distribution parameters - doing the inverse inverts it!
beta_attract = 1
alpha_threshold = 1
beta_threshold = 1
delta_t = 0.1  # time step size
time_steps_max = int(
    total_time / delta_t
)  # number of time steps max, will stop if culture converges
culture_momentum = 5
set_seed = 1  ##reproducibility INTEGER
#np.random.seed(set_seed)
#phi_list = np.linspace(0.9, 1, num=M)
#carbon_emissions = np.linspace(0.5, 1, num=M)
#np.random.shuffle(phi_list)
#np.random.shuffle(carbon_emissions)
phi_list_lower,phi_list_upper = 0.8,1
# print("phi carbon = ",phi_list,carbon_emissions)
prob_rewire = 0.1  # re-wiring probability?
culture_momentum = 1  # real time over which culture is calculated for INTEGER
learning_error_scale = 0.01  # 1 standard distribution is 2% error

#params = [opinion_dynamics,save_data, time_steps_max, delta_t,phi_list_lower,phi_list_upper,N,M,K,prob_rewire,set_seed,culture_momentum,learning_error_scale,alpha_attract,beta_attract,alpha_threshold,beta_threshold]


params = {
    "opinion_dynamics": opinion_dynamics,
    "save_data": save_data, 
    "time_steps_max": time_steps_max, 
    "delta_t": delta_t,
    "phi_list_lower": phi_list_lower,
    "phi_list_upper": phi_list_upper,
    "N": N,
    "M": M,
    "K": K,
    "prob_rewire": prob_rewire,
    "set_seed": set_seed,
    "culture_momentum": culture_momentum,
    "learning_error_scale": learning_error_scale,
    "alpha_attract": alpha_attract,
    "beta_attract": beta_attract,
    "alpha_threshold": alpha_threshold,
    "beta_threshold": beta_threshold
}

params_name = [#THOSE USEd to create the save list?
    opinion_dynamics,
    time_steps_max,
    M,
    N,
    delta_t,
    K,
    prob_rewire,
    set_seed,
    culture_momentum,
    learning_error_scale,
    alpha_attract,
    beta_attract,
    alpha_threshold,
    beta_threshold,
]

# SAVING DATA
# THINGS TO SAVE
data_save_behaviour_array_list = ["value", "attract", "threshold"]
data_save_individual_list = ["culture", "carbon_emissions"]
data_save_network_list = [
    "time",
    "cultural_var",
    "total_carbon_emissions",
    "weighting_matrix_convergence",
    "average_culture",
    "min_culture",
    "max_culture",
]  # "carbon_price"
data_save_network_array_list = [
    "weighting_matrix",
    "behavioural_attract_matrix",
    "social_component_matrix",
]
to_save_list = [
    data_save_behaviour_array_list,
    data_save_individual_list,
    data_save_network_list,
    data_save_network_array_list,
]

# LOAD DATA
paramList = [
    "opinion_dynamics",
    "time_steps_max",
    "M",
    "N",
    "delta_t",
    "K",
    "prob_rewire",
    "set_seed",
    "culture_momentum",
    "learning_error_scale",
    "alpha_attract",
    "beta_attract",
    "alpha_threshold",
    "beta_threshold"
]
loadBooleanCSV = [
    "individual_culture",
    "individual_carbon_emissions",
    "network_total_carbon_emissions",
    "network_time",
    "network_cultural_var",
    "network_weighting_matrix_convergence",
    "network_average_culture",
    "network_min_culture",
    "network_max_culture",
]  # "network_cultural_var",,"network_carbon_price"
loadBooleanArray = [
    "network_weighting_matrix",
    "network_social_component_matrix",
    "network_behavioural_attract_matrix",
    "behaviour_value",
    "behaviour_threshold",
    "behaviour_attract",
]
# "carbon_price_policy_start""culture_var_min","culture_div","nu", "eta"

###PLOT STUFF
nrows_behave = 1
ncols_behave = M
node_size = 50
cmap = LinearSegmentedColormap.from_list("BrownGreen", ["sienna", "white", "olivedrab"])
cmap_weighting = "Reds"
fps = 5
interval = 300
layout = "circular"
round_dec = 2

nrows = 2
ncols = 3

frame_num = ncols * nrows - 1
log_norm = SymLogNorm(
    linthresh=0.15, linscale=1, vmin=-1.0, vmax=1.0, base=10
)  # this works at least its correct
scale_factor = 100
bin_num = 1000
num_counts = 100000
dpi_save = 1200

RUN = True
PLOT = True
SHOW_PLOT = True

if __name__ == "__main__":

    if RUN == False:
        FILENAME = "results/network_100_20_0.2_1001_0.01_5_3_0.01_0_1_1_2_8_8_2_0.02"
    else:
        # start_time = time.time()
        # print("start_time =", time.ctime(time.time()))
        ###RUN MODEL
        print("start_time =", time.ctime(time.time()))
        FILENAME = run(params, to_save_list, params_name)
        # print ("RUN time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))

    if PLOT:

        start_time = time.time()
        print("start_time =", time.ctime(time.time()))

        dataName = FILENAME + "/Data"
        Data = loadData(dataName, loadBooleanCSV, loadBooleanArray)
        Data = get_run_properties(Data, FILENAME, paramList)

        #####BODGES!!
        Data["network_time"] = np.asarray(Data["network_time"])[
            0
        ]  # for some reason pandas does weird shit

        frames_proportion = int(round(len(Data["network_time"]) / 2))
        frames_list = frame_distribution_prints(
            Data["network_time"], scale_factor, frame_num
        )

        ###PLOTS
        plot_beta_distributions(
            FILENAME,
            alpha_attract,
            beta_attract,
            alpha_threshold,
            beta_threshold,
            bin_num,
            num_counts,
            dpi_save,
        )
        plot_culture_timeseries(FILENAME, Data, dpi_save)
        plot_value_timeseries(FILENAME,Data,nrows_behave, ncols_behave,dpi_save)
        #plot_threshold_timeseries(FILENAME,Data,nrows_behave, ncols_behave,dpi_save)
        plot_attract_timeseries(FILENAME, Data, nrows_behave, ncols_behave, dpi_save)
        # plot_carbon_price_timeseries(FILENAME,Data,dpi_save)
        plot_total_carbon_emissions_timeseries(FILENAME, Data, dpi_save)
        plot_av_carbon_emissions_timeseries(FILENAME, Data, dpi_save)
        plot_weighting_matrix_convergence_timeseries(FILENAME, Data, dpi_save)
        plot_cultural_range_timeseries(FILENAME, Data, dpi_save)
        plot_average_culture_timeseries(FILENAME,Data,dpi_save)

        ###PRINTS
        prints_weighting_matrix(FILENAME,Data,cmap_weighting,nrows,ncols,frames_list,round_dec,dpi_save)
        prints_behavioural_matrix(FILENAME,Data,cmap,nrows,ncols,frames_list,round_dec,dpi_save)
        # prints_culture_network(FILENAME,Data,layout,cmap,node_size,nrows,ncols,log_norm,frames_list,round_dec,dpi_save)

        ###ANIMATIONS
        # animate_weighting_matrix(FILENAME,Data,interval,fps,round_dec,cmap_weighting)
        # animate_behavioural_matrix(FILENAME,Data,interval,fps,cmap,round_dec)
        # animate_culture_network(FILENAME,Data,layout,cmap,node_size,interval,fps,log_norm,round_dec)
        # multi_animation(FILENAME,Data,cmap,cmap,layout,node_size,interval,fps,log_norm)
        # multi_animation_alt(FILENAME,Data,cmap,cmap,layout,node_size,interval,fps,log_norm)
        # multi_animation_scaled(FILENAME,Data,cmap,cmap,layout,node_size,interval,fps,scale_factor,frames_proportion,log_norm)

        print(
            "PLOT time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )

        if SHOW_PLOT:
            plt.show()
