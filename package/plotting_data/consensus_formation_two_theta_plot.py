"""

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from package.resources.utility import load_object
from package.resources.plot import (
    plot_consensus_formation,
        plot_consensus_formation_two
)

def main(
    fileName_low_theta = "results/" + "test",
    fileName_high_theta = "results/" + "test",
    dpi_save = 1200,
    latex_bool = 0
    ) -> None: 

    var_identity_low_theta = load_object( fileName_low_theta + "/Data", "var_identity")
    var_no_identity_low_theta = load_object( fileName_low_theta + "/Data", "var_no_identity")
    var_identity_high_theta = load_object( fileName_high_theta + "/Data", "var_identity")
    var_no_identity_high_theta = load_object( fileName_high_theta + "/Data", "var_no_identity")

    property_values_list = load_object(fileName_high_theta + "/Data", "property_values_list")

    base_params_low_theta = load_object(fileName_low_theta + "/Data", "base_params")
    base_params_high_theta = load_object(fileName_high_theta + "/Data", "base_params")

    plot_consensus_formation_two(fileName_low_theta,fileName_high_theta,var_identity_low_theta,var_no_identity_low_theta,var_identity_high_theta,var_no_identity_high_theta,property_values_list,base_params_low_theta["confirmation_bias"],base_params_high_theta["confirmation_bias"], dpi_save, latex_bool = latex_bool)

    plt.show()
