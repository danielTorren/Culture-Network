"""Generate plot of beta distribution
A module that generates a plot of beta distribution according to input data. Useful for visualising
the intial conditions used in the mode for attitudes and thresholds.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
from matplotlib import streamplot
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
import seaborn as sns
from scipy.stats import beta

# constants
alpha_attitude = 1
beta_attitude = 8

alpha_threshold = 0.1
beta_threshold = 6

bin_num = 1000
num_counts = 100000

# modules
def plot_beta_alt(f:str, a_b_combo_list: list):

    fig, ax = plt.subplots()

    x = np.linspace(0,1,100)

    for i in a_b_combo_list:
        y = beta.pdf(x, i[0], i[1])
        ax.plot(x,y, label = r"a = %s, b = %s" % (i[0],i[1]))

    ax.set_xlabel(r"x")
    ax.set_ylabel(r"Probability Density Function")
    ax.legend()

    fig.savefig(f + "%s" % (len(a_b_combo_list)) + ".eps", format="eps")


if __name__ == "__main__":
    FILENAME = "results/plot_beta_distribution"
    
    """
    plot_beta(
        FILENAME,
        alpha_attitude,
        beta_attitude,
        alpha_threshold,
        beta_threshold,
        bin_num,
        num_counts,
    )
    """
    a_b_combo_list = [[5,5],[2,3],[1,1],[0.1,0.1]]

    plot_beta_alt(FILENAME,a_b_combo_list)

    plt.show()
