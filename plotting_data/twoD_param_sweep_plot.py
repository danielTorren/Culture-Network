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
from matplotlib.cm import get_cmap
from resources.plot import (
    double_phase_diagram,
)
from resources.utility import (
    createFolder,
)
from twoD_param_sweep_plot import (
    load_data_av,
    reshape_results_matricies,
)

def main(
    fileName = "results/splitting_eco_warriors_single_add_greens_17_44_05__01_02_2023",
    dpi_save = 1200,
    levels = 10,
) -> None:

    createFolder(fileName)
        
    ######FIX THIS TOO INCLUDE EMISSIONS CHANGE    
    (
        variable_parameters_dict,
        results_emissions,
        results_mu,
        results_var,
        results_coefficient_of_variance,
    ) = load_data_av(fileName)
    ######FIX THIS TOO INCLUDE EMISSIONS CHANGE
    ###PLOTS FOR STOCHASTICALLY AVERAGED RUNS
    (
        matrix_emissions,
        matrix_mu,
        matrix_var,
        matrix_coefficient_of_variance,
    ) = reshape_results_matricies(
        results_emissions,
        results_mu,
        results_var,
        results_coefficient_of_variance,
        variable_parameters_dict["row"]["reps"],
        variable_parameters_dict["col"]["reps"],
        )

    double_phase_diagram(fileName, matrix_emissions, r"Total normalised emissions $E/NM$", "emissions",variable_parameters_dict, get_cmap("Reds"),dpi_save, levels)  

    plt.show()
