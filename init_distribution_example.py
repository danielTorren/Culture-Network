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

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})


# constants
alpha_attitude = 1
beta_attitude = 8

alpha_threshold = 0.1
beta_threshold = 6

bin_num = 1000
num_counts = 100000

# modules
def plot_beta_alt(f:str, a_b_combo_list: list):

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(0,1,100)

    for i in a_b_combo_list:
        y = beta.pdf(x, i[0], i[1])
        ax.plot(x,y, label = r"a = %s, b = %s" % (i[0],i[1]))

    ax.set_xlabel(r"x")
    ax.set_ylabel(r"Probability Density Function")
    ax.legend()

    fig.savefig(f + "%s" % (len(a_b_combo_list)) + ".eps", format="eps")


if __name__ == "__main__":
    fileName = "results/plot_beta_distribution"
    
    """
    plot_beta(
        fileName,
        alpha_attitude,
        beta_attitude,
        alpha_threshold,
        beta_threshold,
        bin_num,
        num_counts,
    )
    """
    a_b_combo_list = [[5,5],[2,3],[1,1],[0.1,0.1]]

    plot_beta_alt(fileName,a_b_combo_list)

    plt.show()
