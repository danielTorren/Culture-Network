import matplotlib.pyplot as plt
import numpy as np
from typing import Union

##inital distribution parameters - doing the inverse inverts it!
alpha_attract = 8
beta_attract = 2

alpha_threshold = 8
beta_threshold = 2

bin_num = 1000
num_counts = 100000


def plot_beta(
    alpha_attract: Union[int, float], beta_attract: Union[int, float], alpha_threshold: Union[int, float], beta_threshold: Union[int, float], bin_num: Union[int, float], num_counts: Union[int, float]
) -> None:

    fig, ax = plt.subplots()

    ax.hist(
        np.random.beta(alpha_attract, beta_attract, num_counts),
        bin_num,
        density=True,
        facecolor="g",
        alpha=0.5,
        histtype="stepfilled",
        label="Attract: alpha = "
        + str(alpha_attract)
        + ", beta = "
        + str(beta_attract),
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


plot_beta(
    alpha_attract, beta_attract, alpha_threshold, beta_threshold, bin_num, num_counts
)
plt.show()
