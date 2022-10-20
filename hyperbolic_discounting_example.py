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

# modules
def calc_discount_list(
    present_discount_factor: float,
    discount_factor: float,
    time_list_beahviours: npt.NDArray,
) -> npt.NDArray:
    """
    Calculate the truncated quasi-hyperbolic discount factor array where the first entry is 1, representing how the present is
    always the most important, but then subsequent terms are exponential decreasing by present bias beta and discount parameter delta

    Parameters
    ----------
    discount_factor: float
        the degree to which each previous time step has a decreasing importance to an individuals memory. Domain = [0,1]
    present_discount_factor: float
        the degree to which a single time step in the past is worth less than the present vs immediate past. The lower this parameter
        the more emphasis is put on the present as the most important moment
    time_list_beahviours: npt.NDArray
        time points used

    Returns
    -------
    discount_list: npt.NDArray
        truncated quasi-hyperbolic discount factor array
    """

    discount_list = present_discount_factor * (discount_factor) ** (
        time_list_beahviours
    )
    discount_list[0] = 1
    return discount_list


def calc_data_discount(
    beta_vals: list, delta_vals: list, time_list: npt.NDArray
) -> tuple[list, list]:
    """
    For different parameters of beta and delta calculate the different discount factor arrays

    Parameters
    ----------
    beta_vals: list
        values of beta  the present bias used in graph
    delta_vals: list
        values of delta the discount parameter used in graph
    time_list: npt.NDArray
        time points used

    Returns
    -------
    const_delta_discount_list: list[npt.NDArray]
        list of time series data of discount factor for the case where discount parameter delta is constant
    const_beta_discount_list; list[npt.NDArray]
        list of time series data of discount factor for the case where present bias beta is constant
    """

    const_delta_discount_list = [
        calc_discount_list(x, delta_vals[0], time_list) for x in beta_vals
    ]
    const_beta_discount_list = [
        calc_discount_list(beta_vals[0], x, time_list) for x in delta_vals
    ]
    return const_delta_discount_list, const_beta_discount_list


def plot_discount_factors_beta_delta(
    f: str,
    const_delta_discount_list: list,
    const_beta_discount_list: list,
    beta_vals: list,
    delta_vals: list,
    time_list: npt.NDArray,
    culture_momentum: float,
    dpi_save: int,
) -> None:
    """
    Plot several distributions for the truncated quasi-hyperbolic discounting factor for different parameter values

    Parameters
    ----------
    f: str
        filename, where plot is saved
    const_delta_discount_list: list[list]
        list of time series data of discount factor for the case where discount parameter delta is constant
    const_beta_discount_list; list[list]
        list of time series data of discount factor for the case where present bias beta is constant
    beta_vals: list
        values of beta  the present bias used in graph
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
    fig, ax = plt.subplots()

    for i in range(len(beta_vals)):
        ax.plot(
            time_list,
            const_delta_discount_list[i],
            linestyle="--",
            label=r"$\beta$ = %s , $\delta$ = %s" % (beta_vals[i], delta_vals[0]),
        )  # bodge so that we dont repeat one of the lines
    for i in range(len(delta_vals) - 1):
        ax.plot(
            time_list,
            const_beta_discount_list[i + 1],
            linestyle="-",
            label=r"$\beta$ = %s , $\delta$ = %s" % (beta_vals[0], delta_vals[i + 1]),
        )

    ax.set_xlabel(r"Time/$\Delta t$")
    ax.set_ylabel(r"$D_s$")
    ax.set_xticks(np.arange(0, -culture_momentum, step=-5))
    ax.legend()

    fig.savefig(f, dpi=dpi_save, format="eps")


if __name__ == "__main__":

    delta_t = 0.5
    culture_momentum = 11
    beta_vals = [0.8, 0.6, 0.4]
    delta_vals = [0.8, 0.7, 0.6, 0.5]

    FILENAME = "results/plot_discount_factors_beta_delta.eps"
    dpi_save = 1200
    steps = int(culture_momentum / delta_t)

    time_list = np.asarray([delta_t * x for x in range(steps)])
    const_delta_discount_list, const_beta_discount_list = calc_data_discount(
        beta_vals, delta_vals, time_list
    )

    time_list_plot = np.asarray(
        [-delta_t * x for x in range(steps)]
    )  # so time is negative, past influences less
    plot_discount_factors_beta_delta(
        FILENAME,
        const_delta_discount_list,
        const_beta_discount_list,
        beta_vals,
        delta_vals,
        time_list_plot,
        culture_momentum,
        dpi_save,
    )

    plt.show()
