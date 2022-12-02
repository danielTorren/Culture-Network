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
import numpy.typing as npt

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


# modules
def plot_discount_factors_delta(
    f: str,
    delta_discount_list: list,
    delta_vals: list,
    time_list: npt.NDArray,
    culture_momentum: float,
    dpi_save: int,
) -> None:
    """
    Plot several distributions for the truncated discounting factor for different parameter values

    Parameters
    ----------
    f: str
        filename, where plot is saved
    const_delta_discount_list: list[list]
        list of time series data of discount factor for the case where discount parameter delta is constant
    delta_vals: list
        values of delta the discount parameter used in graph
    time_list: npt.NDArray
        time points used
    culture_momentum: float
        the number of steps into the past that are considered when individuals consider their identity
    dpi_save: int
        the dpi of image saved

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(len(delta_vals)):
        ax.plot(
            time_list,
            delta_discount_list[i],
            linestyle="--",
            label=r"$\delta$ = %s" % (delta_vals[i]),
        )  # bodge so that we dont repeat one of the lines

    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Discount array, $D_t$")
    ax.set_xticks(np.arange(0, -culture_momentum, step=-20))
    ax.legend()

    fig.savefig(f, dpi=dpi_save, format="eps")


if __name__ == "__main__":

    delta_t = 1
    culture_momentum = 101
    delta_vals = [0.99,0.97,0.95,0.9,0.8]

    fileName = "results/plot_discount_factors_delta.eps"
    dpi_save = 1200
    steps = int(culture_momentum / delta_t)

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
    )

    plt.show()
