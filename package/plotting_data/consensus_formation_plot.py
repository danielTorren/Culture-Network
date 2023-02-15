"""

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from package.resources.utility import load_object
from package.resources.plot import (
    plot_consensus_formation
)

def main(
    fileName = "results/" + "test",
    dpi_save = 1200,
    latex_bool = 0
    ) -> None: 

    var_identity = load_object( fileName + "/Data", "var_identity")
    var_no_identity = load_object( fileName + "/Data", "var_no_identity")
    property_values_list = load_object(fileName + "/Data", "property_values_list")

    plot_consensus_formation(fileName,var_identity,var_no_identity ,property_values_list, dpi_save, latex_bool = latex_bool)

    plt.show()
