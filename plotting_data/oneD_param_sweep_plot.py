"""Run multiple single simulations varying a single parameter
A module that use input data to generate data from multiple social networks varying a single property
between simulations so that the differences may be compared. These multiple runs can either be single 
shot runs or taking the average over multiple runs.

TWO MODES 
    Single parameters can be varied to cover a list of points. This can either be done in SINGLE = True where individual 
    runs are used as the output and gives much greater variety of possible plots but all these plots use the same initial
    seed. Alternatively can be run such that multiple averages of the simulation are produced and then the data accessible 
    is the emissions, mean identity, variance of identity and the coefficient of variance of identity.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from resources.utility import load_object
from resources.plot import (
    live_print_culture_timeseries,
    print_live_intial_culture_networks_and_culture_timeseries,

)

def main(
    fileName = "results/one_param_sweep_single_17_43_28__31_01_2023",
    GRAPH_TYPE = 0,
    node_size = 100,
    round_dec = 2,
    dpi_save = 1200,
    nrows_plot = 2, #leave as 1 for alpha and homophily plots, but change for network!
    ncols_plot = 3,  # due to screen ratio want more cols than rows usually
    ) -> None: 

    cmap = LinearSegmentedColormap.from_list(
        "BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1
    ),
    norm_zero_one = Normalize(vmin=0, vmax=1),

    ############################

    data_list = load_object(fileName + "/Data", "data_list")
    property_varied = load_object(fileName + "/Data", "property_varied")
    property_varied_title = load_object(fileName + "/Data", "property_varied_title")
    title_list = load_object(fileName + "/Data", "title_list")
    property_values_list = load_object(fileName + "/Data", "property_values_list")


    if GRAPH_TYPE == 0:
        #FOR POLARISATION A,B PLOT
        live_print_culture_timeseries(fileName, data_list, property_varied, title_list, nrows_plot, ncols_plot, dpi_save)
    elif GRAPH_TYPE == 1:
        ###############################
        #FOR HOMOPHILY PLOT
        layout = ["circular","circular", "circular"]
        print_live_intial_culture_networks_and_culture_timeseries(fileName, data_list, dpi_save, property_values_list, property_varied_title, ncols_plot, layout, norm_zero_one, cmap, node_size,round_dec)
    plt.show()
