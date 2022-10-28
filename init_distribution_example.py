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

# constants
alpha_attitude = 1
beta_attitude = 8

alpha_threshold = 0.1
beta_threshold = 6

bin_num = 1000
num_counts = 100000

# modules
def plot_beta(
    f: streamplot,
    alpha_attitude: Union[int, float],
    beta_attitude: Union[int, float],
    alpha_threshold: Union[int, float],
    beta_threshold: Union[int, float],
    bin_num: Union[int, float],
    num_counts: Union[int, float],
) -> None:

    """
    Produce histogram of two beta distributions to visually represent the initial distribution of attitude and threshold values

    Parameters
    ----------
    f: str
        filename, where plot is saved
    alpha_attitude: Union[int, float]
        the alpha or a value used for the beta distribution for the attitude initial values
    beta_attitude: Union[int, float]
        the beta or b value used for the beta distribution for the attitude initial values
    alpha_threshold: Union[int, float]
        the alpha or a value used for the beta distribution for the threshold initial values
    beta_threshold: Union[int, float]
        the beta or b value used for the beta distribution for the threshold initial values
    bin_num: Union[int, float]
        size of bin
    num_counts: Union[int, float]
        number of points drawn

    Returns
    -------
    None
    """

    fig, ax = plt.subplots()

    ax.hist(
        np.random.beta(alpha_attitude, beta_attitude, num_counts),
        bin_num,
        density=True,
        facecolor="g",
        alpha=0.5,
        histtype="stepfilled",
        label=r"Attitude: $\alpha$ = "
        + str(alpha_attitude)
        + r", $\beta$ = "
        + str(beta_attitude),
    )
    ax.hist(
        np.random.beta(alpha_threshold, beta_threshold, num_counts),
        bin_num,
        density=True,
        facecolor="b",
        alpha=0.5,
        histtype="stepfilled",
        label=r"Threshold: $\alpha$ = "
        + str(alpha_threshold)
        + r", $\beta$ = "
        + str(beta_threshold),
    )
    ax.set_xlabel(r"x")
    ax.set_ylabel(r"PDF")
    ax.legend()

    fig.savefig(f, format="eps")


if __name__ == "__main__":
    FILENAME = "results/plot_beta_distribution.eps"
    plot_beta(
        FILENAME,
        alpha_attitude,
        beta_attitude,
        alpha_threshold,
        beta_threshold,
        bin_num,
        num_counts,
    )

    plt.show()
