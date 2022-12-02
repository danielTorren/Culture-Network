"""Run multiple simulations varying two parameters
A module that use input data to generate data from multiple social networks varying two properties
between simulations so that the differences may be compared. These multiple runs can either be single 
shot runs or taking the average over multiple runs. Useful for comparing the influences of these parameters
on each other to generate phase diagrams.

TWO MODES 
    The two parameters can be varied covering a 2D plane of points. This can either be done in SINGLE = True where individual 
    runs are used as the output and gives much greater variety of possible plots but all these plots use the same initial
    seed. Alternatively can be run such that multiple averages of the simulation are produced and then the data accessible 
    is the emissions, mean identity, variance of identity and the coefficient of variance of identity.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
import json
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import get_cmap
from resources.plot import (
    plot_compare_av_culture_seed,
    plot_compare_time_culture_seed,
    plot_network_emissions_timeseries_no_culture,
    plot_compare_time_behaviour_culture_seed,
    plot_behaviorual_emissions_timeseries_no_culture,
    cluster_estimation,
)
from resources.utility import (
    createFolder,
    save_object,
    load_object
)
from resources.run import parallel_run
import numpy as np

def produce_param_list(base_params: dict, property_list: list, property: str) -> list[dict]: 

    base_params_list = []
    for i in property_list:
        base_params[property] = i
        base_params_list.append(
            base_params.copy()
        )  # have to make a copy so that it actually appends a new dict and not just the location of the base_params dict
    return base_params_list

###PLOT STUFF
dpi_save = 1200
round_dec = 2
cmap_weighting = "Reds"
cmap_edge = get_cmap("Greys")
fps = 5
interval = 50
layout = "circular"
node_size = 50
cmap = LinearSegmentedColormap.from_list(
    "BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1
)
norm_zero_one = Normalize(vmin=0, vmax=1)

# run bools
RUN = 1 # run or load in previously saved data

colour_list = ["red", "green", "blue"]

if __name__ == "__main__":
    if RUN:
        # load base base_params
        f_base_params = open("constants/base_params.json")
        base_params = json.load(f_base_params)
        f_base_params.close()
        base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])

        fileName = "results/culture_and_no_culture_%s_%s_%s" % (
            str(base_params["N"]),
            str(base_params["time_steps_max"]),
            str(base_params["M"]),
        )
        print("fileName: ", fileName)
        createFolder(fileName)
        
        
        #No culture, behaviours are independant
        base_params["alpha_change"] = 2.0
        property_varied_no_culture = "set_seed"
        property_values_list_no_culture = [1,2,3]
        params_list_no_culture = produce_param_list(base_params, property_values_list_no_culture, property_varied_no_culture)
        data_no_culture = parallel_run(params_list_no_culture)  # better if a Multiple of 4

        #culture, behaviours are dependant via identity
        base_params["alpha_change"] = 1.0
        property_varied_culture = "set_seed"
        property_values_list_culture = [1,2,3]
        params_list_culture = produce_param_list(base_params, property_values_list_culture, property_varied_culture)
        data_culture = parallel_run(params_list_culture)  # better if a Multiple of 4

        #####
        #save stuff
        save_object(base_params, fileName + "/Data", "base_params")
        save_object(data_no_culture, fileName + "/Data", "data_no_culture")
        save_object(data_culture, fileName + "/Data", "data_culture")
        save_object(property_values_list_no_culture, fileName + "/Data", "property_values_list_no_culture")
        save_object(property_varied_no_culture, fileName + "/Data", "property_varied_no_culture,")
        save_object(property_values_list_culture, fileName + "/Data", "property_values_list_culture")
        save_object(property_varied_culture, fileName + "/Data", "property_varied_culture")
    else:
        fileName = "results/culture_and_no_culture_200_300_3"

        base_params = load_object(fileName + "/Data", "base_params")
        data_no_culture = load_object(fileName+ "/Data", "data_no_culture")
        data_culture = load_object( fileName  + "/Data", "data_culture")
        property_values_list_no_culture = load_object( fileName  + "/Data", "property_values_list_no_culture")
        property_varied_no_culture = load_object( fileName  + "/Data", "property_varied_no_culture,")
        property_values_list_culture = load_object(fileName  + "/Data", "property_values_list_culture")
        property_varied_culture = load_object( fileName  + "/Data", "property_varied_culture")

    #PLOTS
    #bandwidth = 1
    #cluster_estimation(data_culture[0],bandwidth)

    plot_compare_av_culture_seed(fileName, data_no_culture,data_culture, 1, 2, dpi_save,property_values_list_no_culture, property_varied_no_culture, property_values_list_culture, property_varied_culture)
    plot_compare_time_culture_seed(fileName, data_no_culture,data_culture, 1, 2, dpi_save,property_values_list_no_culture, property_varied_no_culture, property_values_list_culture, property_varied_culture, colour_list)
    
    plot_compare_time_behaviour_culture_seed(fileName, data_no_culture,data_culture, 2, 3, dpi_save,colour_list)#params["M"]
    plot_network_emissions_timeseries_no_culture(fileName, data_no_culture,data_culture, dpi_save,colour_list)
    plot_behaviorual_emissions_timeseries_no_culture(fileName, data_no_culture,data_culture, dpi_save,colour_list)

    ### PLOTS 
    plt.show()