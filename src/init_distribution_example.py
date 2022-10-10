"""Generate plot of beta distribution
A module that generates a plot of beta distribution according to input data. Useful for visualising
the intial conditions used in the mode for attitudes and thresholds.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

#imports
import matplotlib.pyplot as plt
import numpy as np
from typing import Union

#constants
alpha_attitude = 3
beta_attitude = 2

alpha_threshold = 3
beta_threshold = 2

bin_num = 1000
num_counts = 100000

#modules
def plot_beta(
    alpha_attitude: Union[int, float], beta_attitude: Union[int, float], alpha_threshold: Union[int, float], beta_threshold: Union[int, float], bin_num: Union[int, float], num_counts: Union[int, float]
) -> None:

    fig, ax = plt.subplots()

    ax.hist(
        np.random.beta(alpha_attitude, beta_attitude, num_counts),
        bin_num,
        density=True,
        facecolor="g",
        alpha=0.5,
        histtype="stepfilled",
        label="attitude: alpha = "
        + str(alpha_attitude)
        + ", beta = "
        + str(beta_attitude),
    )
    ax.hist(
        np.random.beta(alpha_threshold, beta_threshold, num_counts),
        bin_num,
        density=True,
        facecolor="b",
        alpha=0.5,
        histtype="stepfilled",
        label="Threshold: alpha = "
        + str(alpha_threshold)
        + ", beta = "
        + str(beta_threshold),
    )
    ax.set_xlabel(r"x")
    ax.set_ylabel(r"PDF")
    ax.legend()

if __name__ == "__main__":
    plot_beta(
        alpha_attitude, beta_attitude, alpha_threshold, beta_threshold, bin_num, num_counts
    )
    
    plt.show()