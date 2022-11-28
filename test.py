"""Runs a single simulation to produce data which is saved and plotted 
A module that use dictionary of data for the simulation run. The single shot simualtion is run
for a given intial set seed. The desired plots are then produced and saved.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""
# imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, SymLogNorm
from matplotlib.cm import get_cmap
import numpy as np
from resources.utility import (
    createFolder, 
    save_object, 
    load_object,
)
"""
from resources.plot import (
    print_culture_density_timeseries_multi
)
"""
from resources.run import parallel_run

# FOR FILENAME
params_name = [  # THOSE USEd to create the save list?
    "time_steps_max",
    "M",
    "N",
    "delta_t",
    "K",
    "prob_rewire",
    "set_seed",
    "learning_error_scale",
    "culture_momentum_real",
    
]

###PLOT STUFF
node_size = 50
cmap = LinearSegmentedColormap.from_list(
    "BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1
)

# norm_neg_pos = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=-1, vmax=1))
norm_neg_pos = Normalize(vmin=-1, vmax=1)
norm_zero_one = Normalize(vmin=0, vmax=1)
log_norm = SymLogNorm(
    linthresh=0.15, linscale=1, vmin=-1.0, vmax=1.0, base=10
)  # this works at least its correct

# log_norm = SymLogNorm(linthresh=0.15, linscale=1, vmin=-1.0, vmax=1.0, base=10)  # this works at least its correct
cmap_weighting = "Reds"
cmap_edge = get_cmap("Greys")
fps = 5
interval = 50
layout = "circular"
round_dec = 2

nrows = 2
ncols = 3

alpha_quick, alpha_normal, alpha_lagard = 0.9, 0.7, 0.9
colour_quick, colour_normal, colour_lagard = "indianred", "grey", "cornflowerblue"

# print("time_steps_max", time_steps_max)
frame_num = ncols * nrows - 1


dpi_save = 2000

min_k, max_k = (
    2,
    10,
)  # N - 1# Cover all the possible bases with the max k though it does slow it down
alpha_val = 0.25
size_points = 5
min_culture_distance = 0.5

RUN = 1
PLOT = 1
SHOW_PLOT = 1

params = {
    "total_time": 2000,
    "delta_t": 1.0,
    "compression_factor": 10,
    "save_data": 1, 
    "alpha_change" : 1.0,
    "degroot_aggregation": 1,
    "averaging_method": "Arithmetic",
    "phi_lower": 0.005,
    "phi_upper": 0.01,
    "N": 200,
    "M": 3,
    "K": 20,
    "prob_rewire": 0.1,
    "set_seed": 1,
    "seed_list": [1,2,3,4,5,6,7,8,9,10],
    "culture_momentum_real": 100,
    "learning_error_scale": 0.02,
    "discount_factor": 0.95,
    "homophily": 0.95,
    "homophilly_rate" : 1,
    "confirmation_bias": 30,
    "a_attitude": 0.5,
    "b_attitude": 0.5,
    "a_threshold": 1,
    "b_threshold": 1,
    "action_observation_I": 0.0,
    "action_observation_S": 0.0,
    "green_N": 0,
    "network_structure": "small_world"

}

if __name__ == "__main__":
    params["time_steps_max"] = int(params["total_time"] / params["delta_t"])

    fileName = "results/test_density"
    createFolder(fileName)#

    RUN = 0

    confirmation_bias_list = [-10,0.0,10,20,40]
    polarisation_list = [10,5,1,0.5,0.1]
    title_list = []

    Data_list = []
    params_list = []
    for i in confirmation_bias_list:
        params["confirmation_bias"] = i
        for j in polarisation_list:
            params["a_attitude"] = j
            params["b_attitude"] = j
            for v in  params["seed_list"]: 
                params["set_seed"] = v
                params_list.append(params.copy())
            title_list.append(r"$theta=%s$, a,b=%s" % (str(i), str(j)))

    ##print(len(params_list))
    ##quit()
        
        
    if RUN:
        
        #i think i can use this everywhere
        Data_list = parallel_run(params_list)
        ys_array_list = []
        row_len = len(confirmation_bias_list)
        for i in range(len(confirmation_bias_list)):
            for j in range(len(polarisation_list)):
                pos = i*(row_len) + j    
                ys_list = []
                for v in  params["seed_list"]:
                    ys = [n.history_culture for n in Data_list[pos].agent_list]
                    ys_list = ys_list + ys
                ys_array_list.append(np.asarray(ys_list))

        x_list = np.asarray(Data_list[0].history_time)

        save_object(Data_list, fileName + "/Data", "Data_list")
        save_object(ys_array_list, fileName + "/Data", "ys_array_list")
        
    else:

        x_list = np.arange(0,params["total_time"] + params["compression_factor"], params["compression_factor"])
        ys_array_list = load_object(fileName + "/Data", "ys_array_list")
        

    ny = 500 
    #print_culture_density_timeseries_multi(fileName, ys_array_list, x_list, title_list, len(confirmation_bias_list), len(polarisation_list), dpi_save, ny)


    plt.show()

