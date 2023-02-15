"""Generate plot of truncated quasi-hyperbolic discounting
A module that generates data for quasi-hyperbolic discounting this is then trucated to produce a
moving average of agent memory. Used for illustrative purporses to explain how individuals have a
cultural momentum which slows identity change.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
from package.resources.plot import plot_discount_factors_delta

def main(
    delta_t = 1,
    culture_momentum = 101,
    delta_vals = [0.99,0.97,0.95,0.9,0.8],
    dpi_save = 1200,
    latex_bool = 0
    ) -> None: 

    steps = int(culture_momentum / delta_t)
    fileName = "results/plot_discount_factors_delta.eps"

    time_list = np.asarray([delta_t * x for x in range(steps)])
    delta_discount_list = []
    for i in delta_vals:
        discount_list = (i)**(time_list)
        discount_list[0] = 1
        delta_discount_list.append(discount_list)

    time_list_plot = np.asarray([-delta_t * x for x in range(steps)])  # so time is negative, past influences less
    
    plot_discount_factors_delta(
        fileName,
        delta_discount_list,
        delta_vals,
        time_list_plot,
        culture_momentum,
        dpi_save,
        latex_bool = latex_bool
    )

    plt.show()
