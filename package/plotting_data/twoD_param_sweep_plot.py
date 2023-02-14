"""Plot multiple simulations varying two parameters

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from package.resources.plot import (
    double_phase_diagram
)
from package.resources.utility import (
    load_object
)

def main(
    fileName = "results/splitting_eco_warriors_single_add_greens_17_44_05__01_02_2023",
    dpi_save = 1200,
    levels = 10,
) -> None:
        
    variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
    results_emissions = load_object(fileName + "/Data", "results_emissions")
    #results_mu =  load_object(fileName + "/Data", "results_mu")
    #results_var =  load_object(fileName + "/Data", "results_var")
    #results_coefficient_of_variance = load_object(fileName + "/Data","results_coefficient_of_variance")
    #results_emissions_change = load_object( fileName + "/Data", "results_emissions_change")

    matrix_emissions = results_emissions.reshape((variable_parameters_dict["row"]["reps"], variable_parameters_dict["col"]["reps"]))

    double_phase_diagram(fileName, matrix_emissions, r"Total normalised emissions $E/NM$", "emissions",variable_parameters_dict, get_cmap("Reds"),dpi_save, levels)  

    plt.show()
