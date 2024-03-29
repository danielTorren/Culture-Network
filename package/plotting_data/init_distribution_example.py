"""Generate plot of beta distribution
A module that generates a plot of beta distribution according to input data. Useful for visualising
the initial conditions used in the mode for attitudes and thresholds.



Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
from package.resources.plot import plot_beta_alt

# modules

def main(
    a_b_combo_list = [[2,5],[2,2],[5,2]],
    latex_bool = 0
    ) -> None: 

    fileName = "results/plot_beta_distribution"
    
    plot_beta_alt(fileName,a_b_combo_list,latex_bool = latex_bool)

    plt.show()
