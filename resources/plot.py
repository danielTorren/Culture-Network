"""Plot simulation results
A module that use input data or social network object to produce plots for analysis.
These plots also include animations or phase diagrams.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import networkx as nx
from networkx import Graph
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize, LinearSegmentedColormap, SymLogNorm
from matplotlib.collections import LineCollection
from matplotlib.cm import get_cmap
from typing import Union

from resources.network import Network


# modules
#####RUNPLOT PLOTS - SINGLE NETWORK
def plot_culture_timeseries(fileName, Data, dpi_save):
    fig, ax = plt.subplots()
    y_title = "Identity"

    for v in Data.agent_list:
        ax.plot(np.asarray(Data.history_time), np.asarray(v.history_culture))
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_ylim(0, 1)

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_culture_timeseries.eps"
    fig.savefig(f, dpi=dpi_save, format="eps")


def plot_network_timeseries(
    FILENAME: str, Data: Network, y_title: str, property: str, dpi_save: int
):

    fig, ax = plt.subplots()
    data = eval("Data.%s" % property)

    # bodge
    ax.plot(Data.history_time, data)
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    plotName = FILENAME + "/Plots"
    f = plotName + "/" + property + "_timeseries.eps"
    fig.savefig(f, dpi=dpi_save, format="eps")


def plot_cultural_range_timeseries(FILENAME: str, Data: DataFrame, dpi_save: int):
    y_title = "Identity variance"
    property = "history_var_culture"
    plot_network_timeseries(FILENAME, Data, y_title, property, dpi_save)


def plot_weighting_matrix_convergence_timeseries(
    FILENAME: str, Data: DataFrame, dpi_save: int
):
    y_title = "Change in Agent Link Strength"
    property = "history_weighting_matrix_convergence"
    plot_network_timeseries(FILENAME, Data, y_title, property, dpi_save)


def plot_total_carbon_emissions_timeseries(
    FILENAME: str, Data: DataFrame, dpi_save: int
):
    y_title = "Carbon Emissions"
    property = "history_total_carbon_emissions"
    plot_network_timeseries(FILENAME, Data, y_title, property, dpi_save)


def plot_green_adoption_timeseries(FILENAME: str, Data: DataFrame, dpi_save: int):
    y_title = "Green adoption %"
    property = "history_green_adoption"

    plot_network_timeseries(FILENAME, Data, y_title, property, dpi_save)


def plot_average_culture_timeseries(FILENAME: str, Data: DataFrame, dpi_save: int):
    y_title = "Average identity"
    property = "history_average_culture"

    plot_network_timeseries(FILENAME, Data, y_title, property, dpi_save)


def plot_cum_link_change_per_agent(fileName: str, Data: Network, dpi_save: int):

    fig, ax = plt.subplots()
    y_title = "Cumulative total link strength change per agent"

    cumulative_link_change = np.cumsum(
        np.asarray(Data.history_weighting_matrix_convergence) / Data.N
    )

    ax.plot(
        np.asarray(Data.history_time),
        cumulative_link_change,
    )
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_cum_link_change_per_agent.eps"
    fig.savefig(f, dpi=dpi_save, format="eps")


def plot_individual_timeseries(
    fileName: str,
    Data: Network,
    y_title: str,
    property: str,
    dpi_save: int,
    ylim_low: int,
):
    fig, axes = plt.subplots(nrows=1, ncols=Data.M, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):
        for v in range(len(Data.agent_list)):
            data_ind = np.asarray(eval("Data.agent_list[%s].%s" % (str(v), property)))
            ax.plot(np.asarray(Data.history_time), data_ind[:, i])
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_ylim(ylim_low, 1)

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_%s_timeseries.eps" % property
    fig.savefig(f, dpi=dpi_save, format="eps")


def plot_value_timeseries(FILENAME: str, Data: DataFrame, dpi_save: int):
    y_title = "Behavioural value, B"
    property = "history_behaviour_values"
    ylim_low = -1

    plot_individual_timeseries(FILENAME, Data, y_title, property, dpi_save, ylim_low)


def plot_attitude_timeseries(FILENAME: str, Data: DataFrame, dpi_save: int):
    y_title = "Behavioural attiude, A"
    property = "history_behaviour_attitudes"
    ylim_low = 0

    plot_individual_timeseries(FILENAME, Data, y_title, property, dpi_save, ylim_low)


def plot_threshold_timeseries(FILENAME: str, Data: DataFrame, dpi_save: int):
    y_title = "Behavioural threshold, T"
    property = "history_behaviour_thresholds"
    ylim_low = 0

    plot_individual_timeseries(FILENAME, Data, y_title, property, dpi_save, ylim_low)


#################################################################################################################


def live_print_culture_timeseries(
    fileName, Data_list, property_varied, title_list, nrows, ncols, dpi_save
):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    y_title = "Identity"

    for i, ax in enumerate(axes.flat):
        for v in Data_list[i].agent_list:
            ax.plot(
                np.asarray(Data_list[i].history_time), np.asarray(v.history_culture)
            )

        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title(title_list[i])
        # ax.set_xlim(0,1)
        ax.set_ylim(0, 1)
        # ax.axvline(culture_momentum, color='r',linestyle = "--")

    plt.tight_layout()

    plotName = fileName + "/Prints"
    f = plotName + "/live_plot_culture_timeseries_%s.eps" % property_varied
    fig.savefig(f, dpi=dpi_save, format="eps")


def live_print_culture_timeseries_with_weighting(
    fileName, Data_list, property_varied, title_list, nrows, ncols, dpi_save, cmap
):

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(14, 7), constrained_layout=True
    )

    y_title = "Identity"

    for i in range(ncols):
        for v in Data_list[i].agent_list:
            axes[0][i].plot(
                np.asarray(Data_list[i].history_time), np.asarray(v.history_culture)
            )

        axes[0][i].set_xlabel(r"Time")
        axes[0][i].set_ylabel(r"%s" % y_title)
        axes[0][i].set_title(title_list[i])
        axes[0][i].set_ylim(0, 1)

        axes[1][i].matshow(
            Data_list[i].history_weighting_matrix[-1],
            cmap=cmap,
            norm=Normalize(vmin=0, vmax=1),
            aspect="auto",
        )
        axes[1][i].set_xlabel(r"n")
        axes[1][i].set_ylabel(r"k")
        # print("matrix", Data_list[i].history_weighting_matrix[-1])
        # print("alpha_change",Data_list[i].alpha_change)
    # colour bar axes
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1)),
        ax=axes.ravel().tolist(),
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label(r"$\alpha_{n,k}$")

    plotName = fileName + "/Prints"
    f = (
        plotName
        + "/lowres_live_print_culture_timeseries_with_weighting_%s.png"
        % property_varied
    )
    fig.savefig(f, dpi=dpi_save, format="png")
    # fig.savefig(f, dpi=dpi_save,format='eps')


def live_print_culture_timeseries_vary(
    fileName,
    Data_list,
    property_varied_row,
    property_varied_col,
    title_list,
    nrows,
    ncols,
    dpi_save,
):

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(14, 7), constrained_layout=True
    )

    y_title = "Identity"

    for i, ax in enumerate(axes.flat):
        for v in Data_list[i].agent_list:
            ax.plot(
                np.asarray(Data_list[i].history_time), np.asarray(v.history_culture)
            )

        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title(title_list[i])
        # ax.set_xlim(0,1)
        ax.set_ylim(0, 1)
        # ax.axvline(culture_momentum, color='r',linestyle = "--")

    plotName = fileName + "/Prints"
    f = plotName + "/live_print_culture_timeseries_vary_%s_and_%s" % (
        property_varied_row,
        property_varied_col,
    )
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def live_multirun_diagram_mean_coefficient_variance(
    fileName,
    Data_list,
    property_varied,
    property_varied_values,
    property_title,
    cmap,
    dpi_save,
    norm_zero_one,
):

    fig, ax = plt.subplots(figsize=(14, 7))

    x_data = [i.average_culture for i in Data_list]
    y_data = [i.std_culture / i.average_culture for i in Data_list]
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\sigma /\mu$")

    colour_adjust = norm_zero_one(property_varied_values)
    scat_colours = cmap(colour_adjust)

    ax.scatter(x_data, y_data, s=60, c=scat_colours, edgecolors="black", linewidths=1)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(
            cmap=cmap,
            norm=Normalize(
                vmin=min(property_varied_values), vmax=max(property_varied_values)
            ),
        ),
        ax=ax,
        location="right",
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label(r"%s" % (property_title))

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/live_plot_diagram_mean_coefficient_variance_%s_and_%s.eps" % (
        property_varied,
        len(property_varied_values),
    )
    fig.savefig(f, dpi=dpi_save, format="eps")


def live_average_multirun_diagram_mean_coefficient_variance(
    fileName,
    mean_data,
    coefficient_variance_data,
    property_varied,
    property_varied_values,
    property_title,
    cmap,
    dpi_save,
    norm,
):

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\sigma /\mu$")

    colour_adjust = norm(property_varied_values)
    scat_colours = cmap(colour_adjust)

    ax.scatter(
        mean_data,
        coefficient_variance_data,
        s=60,
        c=scat_colours,
        edgecolors="black",
        linewidths=1,
    )

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=ax,
        location="right",  # Normalize(vmin = min(property_varied_values), vmax=max(property_varied_values))
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label(r"%s" % (property_title))

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/live_plot_diagram_mean_coefficient_variance_%s_and_%s" % (
        property_varied,
        len(property_varied_values),
    )
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def live_average_multirun_n_diagram_mean_coefficient_variance(
    fileName,
    combined_data,
    variable_parameters_dict,
    dpi_save,
):

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)  #
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\sigma /\mu$")

    for i in variable_parameters_dict.keys():
        colour_adjust = variable_parameters_dict[i]["norm"](
            variable_parameters_dict[i]["vals"]
        )
        scatter_colours = variable_parameters_dict[i]["cmap"](colour_adjust)
        ax.scatter(
            combined_data[i]["mean_data"],
            combined_data[i]["coefficient_variance_data"],
            marker=variable_parameters_dict[i]["marker"],
            s=60,
            c=scatter_colours,
            edgecolors="black",
            linewidths=1,
        )

        cbar = fig.colorbar(
            plt.cm.ScalarMappable(
                cmap=get_cmap(variable_parameters_dict[i]["cmap"]),
                norm=variable_parameters_dict[i]["norm"],
            ),
            ax=ax,
            location=variable_parameters_dict[i]["cbar_loc"],
            aspect=60,
            pad=0.05,
        )
        cbar.set_label(r"%s" % (variable_parameters_dict[i]["title"]))

    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plotName = fileName + "/Plots"
    f = plotName + "/live_plot_diagram_mean_coefficient_variance"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def live_average_multirun_n_diagram_mean_coefficient_variance_cols(
    fileName, combined_data, variable_parameters_dict, dpi_save
):

    fig, axes = plt.subplots(
        figsize=(14, 7),
        constrained_layout=True,
        nrows=1,
        ncols=len(variable_parameters_dict.keys()),
    )  #

    key_list = list(variable_parameters_dict.keys())

    for i, ax in enumerate(axes.flat):
        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel(r"$\sigma /\mu$")
        colour_adjust = variable_parameters_dict[key_list[i]]["norm"](
            variable_parameters_dict[key_list[i]]["vals"]
        )
        scatter_colours = get_cmap(variable_parameters_dict[key_list[i]]["cmap"])(
            colour_adjust
        )
        ax.scatter(
            combined_data[key_list[i]]["mean_data"],
            combined_data[key_list[i]]["coefficient_variance_data"],
            marker=variable_parameters_dict[key_list[i]]["marker"],
            s=60,
            c=scatter_colours,
            edgecolors="black",
            linewidths=1,
        )

        cbar = fig.colorbar(
            plt.cm.ScalarMappable(
                cmap=variable_parameters_dict[key_list[i]]["cmap"],
                norm=variable_parameters_dict[key_list[i]]["norm"],
            ),
            ax=ax,
            location=variable_parameters_dict[key_list[i]]["cbar_loc"],
            aspect=60,
            pad=0.05,
        )
        cbar.set_label(r"%s" % (variable_parameters_dict[key_list[i]]["title"]))

    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plotName = fileName + "/Plots"
    f = plotName + "/live_plot_diagram_mean_coefficient_variance_cols"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def live_average_multirun_double_phase_diagram_mean(
    fileName,
    Z,
    property_varied_row,
    property_varied_values_row,
    property_varied_col,
    property_varied_values_col,
    cmap,
    dpi_save,
    round_dec,
):

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)

    ax.set_xlabel(r"%s" % property_varied_col)
    ax.set_ylabel(r"%s" % property_varied_row)

    ax.set_yscale("log")

    X, Y = np.meshgrid(property_varied_values_col, property_varied_values_row)
    contours = ax.contour(X, Y, Z, colors="black")
    ax.clabel(contours, inline=True, fontsize=8)

    cp = ax.contourf(X, Y, Z, cmap=cmap, alpha=0.5)
    cbar = fig.colorbar(
        cp,
        ax=ax,
    )
    cbar.set_label("Mean idenity")

    plotName = fileName + "/Plots"
    f = plotName + "/live_average_multirun_double_phase_diagram_mean"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def live_average_multirun_double_phase_diagram_mean_alt(
    fileName, Z, variable_parameters_dict, cmap, dpi_save
):

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]

    ax.set_xlabel(r"%s" % col_dict["property"])
    ax.set_ylabel(r"%s" % row_dict["property"])

    if col_dict["divisions"] == "log":
        ax.set_xscale("log")
    if row_dict["divisions"] == "log":
        ax.set_yscale("log")

    X, Y = np.meshgrid(col_dict["vals"], row_dict["vals"])
    contours = ax.contour(X, Y, Z, colors="black")
    ax.clabel(contours, inline=True, fontsize=8)

    cp = ax.contourf(X, Y, Z, cmap=cmap, alpha=0.5)
    cbar = fig.colorbar(
        cp,
        ax=ax,
    )
    cbar.set_label("Mean idenity")

    plotName = fileName + "/Plots"
    f = plotName + "/live_average_multirun_double_phase_diagram_mean_alt"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def live_average_multirun_double_phase_diagram_C_of_V(
    fileName,
    Z,
    property_varied_row,
    property_varied_values_row,
    property_varied_col,
    property_varied_values_col,
    cmap,
    dpi_save,
    round_dec,
):

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)

    ax.set_xlabel(r"%s" % property_varied_col)
    ax.set_ylabel(r"%s" % property_varied_row)

    X, Y = np.meshgrid(property_varied_values_col, property_varied_values_row)
    contours = ax.contour(X, Y, Z, colors="black")
    ax.clabel(contours, inline=True, fontsize=8)

    cp = ax.contourf(X, Y, Z, cmap=cmap, alpha=0.5)
    cbar = fig.colorbar(
        cp,
        ax=ax,
    )
    cbar.set_label("Coefficient of variance of idenity")

    plotName = fileName + "/Plots"
    f = plotName + "/live_average_multirun_double_phase_diagram_C_of_V"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def live_average_multirun_double_phase_diagram_C_of_V_alt(
    fileName, Z, variable_parameters_dict, cmap, dpi_save
):

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]

    ax.set_xlabel(r"%s" % col_dict["property"])
    ax.set_ylabel(r"%s" % row_dict["property"])

    if col_dict["divisions"] == "log":
        ax.set_xscale("log")
    if row_dict["divisions"] == "log":
        ax.set_yscale("log")

    X, Y = np.meshgrid(col_dict["vals"], row_dict["vals"])
    contours = ax.contour(X, Y, Z, colors="black")
    ax.clabel(contours, inline=True, fontsize=8)

    cp = ax.contourf(X, Y, Z, cmap=cmap, alpha=0.5)
    cbar = fig.colorbar(
        cp,
        ax=ax,
    )
    cbar.set_label("Coefficient of variance of idenity")

    plotName = fileName + "/Plots"
    f = plotName + "/live_average_multirun_double_phase_diagram_C_of_V_alt"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def double_phase_diagram(
    fileName, Z, Y_title, Y_param, variable_parameters_dict, cmap, dpi_save
):

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]

    ax.set_xlabel(r"%s" % col_dict["title"])
    ax.set_ylabel(r"%s" % row_dict["title"])

    if col_dict["divisions"] == "log":
        ax.set_xscale("log")
    if row_dict["divisions"] == "log":
        ax.set_yscale("log")

    X, Y = np.meshgrid(col_dict["vals"], row_dict["vals"])
    contours = ax.contour(X, Y, Z, colors="black")
    ax.clabel(contours, inline=True, fontsize=8)

    cp = ax.contourf(X, Y, Z, cmap=cmap, alpha=0.5)
    cbar = fig.colorbar(
        cp,
        ax=ax,
    )
    cbar.set_label(Y_title)

    plotName = fileName + "/Plots"
    f = plotName + "/live_average_multirun_double_phase_diagram_%s" % (Y_param)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def double_phase_diagram_using_meanandvariance(
    fileName, Z, Y_title, Y_param, variable_parameters_dict, cmap, dpi_save
):

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]

    ax.set_ylabel(r"$\sigma ^2$")
    ax.set_xlabel(r"$\mu$")

    @np.vectorize
    def convert_ab_to_mu_var(a, b):
        mu = a / (a + b)
        var = a * b / (((a + b) ** 2) * (a + b + 1))
        return mu, var

    A, B = np.meshgrid(col_dict["vals"], row_dict["vals"])

    MU, VAR = convert_ab_to_mu_var(A, B)

    contours = ax.contour(MU, VAR, Z, colors="black")
    ax.clabel(contours, inline=True, fontsize=8)

    cp = ax.contourf(MU, VAR, Z, cmap=cmap, alpha=0.5)

    norm = Normalize(vmin=cp.cvalues.min(), vmax=cp.cvalues.max())
    # a previous version of this used
    # norm= matplotlib.colors.Normalize(vmin=cs.vmin, vmax=cs.vmax)
    # which does not work any more
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cp.cmap)
    cbar = fig.colorbar(
        sm,
        ax=ax,
    )
    cbar.set_label(Y_title)

    plotName = fileName + "/Plots"
    f = (
        plotName
        + "/live_average_multirun_double_phase_diagram_%s_using_meanandvariance"
        % (Y_param)
    )
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def print_culture_timeseries_vary_array(
    fileName: str,
    Data_array: list[Network],
    property_varied_row,
    property_title_row,
    property_varied_values_row,
    property_varied_col,
    property_title_col,
    property_varied_values_col,
    nrows: int,
    ncols: int,
    dpi_save: int,
):

    y_title = "Identity"

    fig = plt.figure(constrained_layout=True)

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=nrows, ncols=ncols)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f"{property_title_row} = {property_varied_values_row[row]}")

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=ncols)
        for col, ax in enumerate(axs):
            for v in Data_array[row][col].agent_list:
                ax.plot(
                    np.asarray(Data_array[row][col].history_time),
                    np.asarray(v.history_culture),
                )
            ax.set_title(f"{property_title_col} = {property_varied_values_col[col]}")
            # ax.set_ylabel(r"%s" % y_title)
            # ax.set_xlabel(r"Time")
    fig.supxlabel(r"Time")
    fig.supylabel(r"%s" % y_title)

    plotName = fileName + "/Prints"
    f = plotName + "/print_culture_timeseries_vary_array_%s_%s.eps" % (
        property_varied_row,
        property_varied_col,
    )
    fig.savefig(f, dpi=dpi_save, format="eps")


def print_culture_time_series_two_properties(
    FILENAME: str,
    Data_list: list,
    property_varied_values_row: list,
    property_varied_values_column: list,
    property_varied_row: str,
    property_varied_column: str,
    nrows: int,
    ncols: int,
    dpi_save: int,
    round_dec,
):

    y_title = "Indivdiual culture"

    fig = plt.figure(constrained_layout=True)

    subfigs = fig.subfigures(nrows=nrows, ncols=1)
    for row, subfig in enumerate(subfigs):

        subfig.suptitle(f"{property_varied_row} = {property_varied_values_row[row]}")

        axs = subfig.subplots(nrows=1, ncols=ncols)
        for col, ax in enumerate(axs):

            X_train = np.asarray(
                [v.history_culture for v in Data_list[row][col].agent_list]
            )
            time_list = np.asarray(Data_list[row][col].history_time)

            for v in range(int(int(Data_list[row][col].N))):
                ax.plot(time_list, X_train[v])
            ax.axvline(
                Data_list[row][col].culture_momentum_real, color="r", linestyle="--"
            )

            ax.set_title(
                r"{} = {}".format(
                    property_varied_column,
                    round(property_varied_values_column[col], round_dec),
                )
            )

    fig.supxlabel(r"Time")
    fig.supylabel(r"%s" % y_title)

    plotName = FILENAME + "/Prints"
    f = plotName + "/print_culture_time_series_two_properties_{}_{}.eps".format(
        property_varied_row, property_varied_column
    )
    fig.savefig(f, dpi=dpi_save, format="eps")


#######################################################################################################################################################################
"""GENERIC MULTI RUN PLOTS"""


def plot_average_culture_no_range_comparison(
    fileName: str,
    Data_list: list[Network],
    dpi_save: int,
    property_list: list,
    property,
    round_dec,
):
    y_title = "Average Culture"

    fig, ax = plt.subplots()
    ax.set_ylabel(r"%s" % y_title)
    for i in range(len(Data_list)):

        ax.plot(
            np.asarray(Data_list[i].history_time),
            np.asarray(Data_list[i].history_average_culture),
            label="%s = %s" % (property, round(property_list[i], round_dec)),
        )

    ax.set_xlabel(r"Time")

    ax.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/%s_plot_average_culture_no_range_comparison.eps" % (property)
    fig.savefig(f, dpi=dpi_save, format="eps")


def plot_average_culture_comparison(
    fileName: str,
    Data_list: list[Network],
    dpi_save: int,
    property_list: list,
    property,
    round_dec,
):
    y_title = "Average Culture"

    fig, ax = plt.subplots()
    ax.set_ylabel(r"%s" % y_title)
    for i in range(len(Data_list)):
        # print(np.asarray(Data_list[i].history_average_culture))
        culture_min = np.asarray(Data_list[i].history_min_culture)  # bodge
        culture_max = np.asarray(Data_list[i].history_max_culture)  # bodge

        ax.plot(
            np.asarray(Data_list[i].history_time),
            np.asarray(Data_list[i].history_average_culture),
            label="%s = %s" % (property, round(property_list[i], round_dec)),
        )

        ax.fill_between(
            np.asarray(Data_list[i].history_time),
            culture_min,
            culture_max,
            alpha=0.5,
            linewidth=0,
        )
    ax.set_xlabel(r"Time")

    ax.legend()

    plotName = fileName + "/Plots"

    f = plotName + "/%s_comparing_av_cultures.eps" % (property)
    fig.savefig(f, dpi=dpi_save, format="eps")


def plot_carbon_emissions_total_comparison(
    fileName: str,
    Data_list: list[Network],
    dpi_save: int,
    property_list: list,
    property,
    round_dec,
):
    y_title = "Total Emissions"

    fig, ax = plt.subplots()
    ax.set_ylabel(r"%s" % y_title)
    for i in range(len(Data_list)):
        ax.plot(
            np.asarray(Data_list[i].history_time),
            np.asarray(Data_list[i].history_total_carbon_emissions),
            label="%s = %s" % (property, round(property_list[i], round_dec)),
        )
        ax.set_xlabel(r"Time")

    ax.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/%s_comparing_total_emissions.eps" % (property)
    fig.savefig(f, dpi=dpi_save, format="eps")


def plot_weighting_matrix_convergence_comparison(
    fileName: str,
    Data_list: list[Network],
    dpi_save: int,
    property_list: list,
    property,
    round_dec,
):
    y_title = "Weighting matrix convergence"

    fig, ax = plt.subplots()
    ax.set_ylabel(r"%s" % y_title)
    for i in range(len(Data_list)):

        ax.plot(
            np.asarray(Data_list[i].history_time),
            np.asarray(Data_list[i].history_weighting_matrix_convergence),
            label="%s = %s" % (property, round(property_list[i], round_dec)),
        )
        ax.set_xlabel(r"Time")

    ax.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/%s_comparing_weighting_matrix_convergence.eps" % (property)
    fig.savefig(f, dpi=dpi_save, format="eps")


def plot_cum_weighting_matrix_convergence_comparison(
    fileName: str,
    Data_list: list[Network],
    dpi_save: int,
    property_list: list,
    property,
    round_dec,
):
    y_title = "Cumulative weighting matrix convergence"

    fig, ax = plt.subplots()
    ax.set_ylabel(r"%s" % y_title)
    for i in range(len(Data_list)):
        cumulative_link_change = np.cumsum(
            np.asarray(Data_list[i].history_weighting_matrix_convergence)
        )
        ax.plot(
            np.asarray(Data_list[i].history_time),
            cumulative_link_change,
            label="%s = %s" % (property, round(property_list[i], round_dec)),
        )
        ax.set_xlabel(r"Time")
    ax.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/%s_comparing_cum_weighting_matrix_convergence.eps" % (property)
    fig.savefig(f, dpi=dpi_save, format="eps")


def plot_live_link_change_comparison(
    fileName: str,
    Data_list: list[Network],
    dpi_save: int,
    property_list: list,
    property,
    round_dec,
):

    fig, ax = plt.subplots()
    y_title = "Total link strength change"

    for i in range(len(Data_list)):
        ax.plot(
            np.asarray(Data_list[i].history_time),
            np.asarray(Data_list[i].history_weighting_matrix_convergence),
            label="{} = {}".format(property, round(property_list[i], round_dec)),
        )
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
    ax.legend()
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/%s_live_link_change.eps" % (property)
    fig.savefig(f, dpi=dpi_save, format="eps")


def plot_live_cum_link_change_comparison(
    fileName: str,
    Data_list: list[Network],
    dpi_save: int,
    property_list: list,
    property,
    round_dec,
):

    fig, ax = plt.subplots()
    y_title = "Cumulative total link strength change"

    for i in range(len(Data_list)):
        cumulative_link_change = np.cumsum(
            np.asarray(Data_list[i].history_weighting_matrix_convergence)
        )

        ax.plot(
            np.asarray(Data_list[i].history_time),
            cumulative_link_change,
            label="{} = {}".format(property, round(property_list[i], round_dec)),
        )
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
    ax.legend()
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/%s_live_cum_link_change.eps" % (property)
    fig.savefig(f, dpi=dpi_save, format="eps")


def plot_live_link_change_per_agent_comparison(
    fileName: str,
    Data_list: list[Network],
    dpi_save: int,
    property_list: list,
    property,
    round_dec,
):

    fig, ax = plt.subplots()
    y_title = "Total link strength change per agent"

    for i in range(len(Data_list)):
        ax.plot(
            np.asarray(Data_list[i].history_time),
            np.asarray(Data_list[i].history_weighting_matrix_convergence)
            / Data_list[i].N,
            label="{} = {}".format(property, round(property_list[i], round_dec)),
        )
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
    ax.legend()
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/%s_live_link_change_per_agent.eps" % (property)
    fig.savefig(f, dpi=dpi_save, format="eps")


def plot_live_cum_link_change_per_agent_comparison(
    fileName: str,
    Data_list: list[Network],
    dpi_save: int,
    property_list: list,
    property,
    round_dec,
):

    fig, ax = plt.subplots()
    y_title = "Cumulative total link strength change per agent"

    for i in range(len(Data_list)):
        cumulative_link_change = np.cumsum(
            np.asarray(Data_list[i].history_weighting_matrix_convergence)
            / Data_list[i].N
        )
        # print("norm",np.asarray(Data_list[i].history_weighting_matrix_convergence))
        # print("cum:", cumulative_link_change)
        ax.plot(
            np.asarray(Data_list[i].history_time),
            cumulative_link_change,
            label="{} = {}".format(property, round(property_list[i], round_dec)),
        )
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
    ax.legend()
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/%s_live_cum_link_change_per_agent.eps" % (property)
    fig.savefig(f, dpi=dpi_save, format="eps")


def print_live_intial_culture_networks(
    fileName: str,
    Data_list: list[Network],
    dpi_save: int,
    property_list: list,
    property,
    nrows: int,
    ncols: int,
    layout: str,
    norm_zero_one,
    cmap,
    node_size,
    round_dec,
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):
        G = nx.from_numpy_matrix(Data_list[i].history_weighting_matrix[0])
        pos_culture_network = prod_pos(layout, G)
        # print(i,ax)
        ax.set_title(r"{} = {}".format(property, round(property_list[i], round_dec)))

        indiv_culutre_list = [v.history_culture[0] for v in Data_list[i].agent_list]
        # print(indiv_culutre_list)
        colour_adjust = norm_zero_one(indiv_culutre_list)
        ani_step_colours = cmap(colour_adjust)

        nx.draw(
            G,
            node_color=ani_step_colours,
            ax=ax,
            pos=pos_culture_network,
            node_size=node_size,
            edgecolors="black",
        )

    # colour bar axes
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm_zero_one), ax=axes.ravel().tolist()
    )
    cbar.set_label("Identity")

    plotName = fileName + "/Prints"
    f = plotName + "/%s_print_intial_culture_networks.eps" % (property)
    fig.savefig(f, dpi=dpi_save, format="eps")


def print_live_intial_culture_networks_and_culture_timeseries(
    fileName: str,
    Data_list: list[Network],
    dpi_save: int,
    property_list: list,
    property,
    ncols: int,
    layout: str,
    norm_zero_one,
    cmap,
    node_size,
    round_dec,
):
    y_title = "Identity"
    fig, axes = plt.subplots(
        nrows=2, ncols=ncols, figsize=(14, 7), constrained_layout=True
    )

    for i in range(ncols):
        #####NETWORK
        G = nx.from_numpy_matrix(Data_list[i].history_weighting_matrix[0])
        pos_culture_network = prod_pos(layout, G)
        # print(i,ax)
        axes[0][i].set_title(
            r"{} = {}".format(property, round(property_list[i], round_dec))
        )

        indiv_culutre_list = [v.history_culture[0] for v in Data_list[i].agent_list]
        # print(indiv_culutre_list)
        colour_adjust = norm_zero_one(indiv_culutre_list)
        ani_step_colours = cmap(colour_adjust)

        nx.draw(
            G,
            node_color=ani_step_colours,
            ax=axes[1][i],
            pos=pos_culture_network,
            node_size=node_size,
            edgecolors="black",
        )

        #####CULTURE TIME SERIES
        for v in Data_list[i].agent_list:
            axes[0][i].plot(
                np.asarray(Data_list[i].history_time), np.asarray(v.history_culture)
            )

        axes[0][i].set_xlabel(r"Time")
        axes[0][i].set_ylabel(r"%s" % y_title)
        axes[0][i].set_ylim(0, 1)

    # colour bar axes
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm_zero_one), ax=axes.ravel().tolist()
    )
    cbar.set_label("Identity")

    plotName = fileName + "/Prints"
    f = (
        plotName
        + "/%s_print_live_intial_culture_networks_and_culture_timeseries.png"
        % (property)
    )
    fig.savefig(f, dpi=dpi_save)

    f_eps = (
        plotName
        + "/%s_print_live_intial_culture_networks_and_culture_timeseries.eps"
        % (property)
    )
    fig.savefig(f_eps, dpi=dpi_save, format="eps")


def prints_init_weighting_matrix(
    fileName: str,
    Data_list: list[Network],
    dpi_save: int,
    nrows: int,
    ncols: int,
    cmap,
    property_list: list,
    property,
    round_dec,
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):
        # print(i)
        mat = ax.matshow(
            Data_list[i].history_weighting_matrix[0],
            cmap=cmap,
            norm=Normalize(vmin=0, vmax=1),
            aspect="auto",
        )
        ax.set_title(r"{} = {}".format(property, round(property_list[i], round_dec)))
        ax.set_xlabel("Agent Link Strength")
        ax.set_ylabel("Agent Link Strength")
    plt.tight_layout()

    # colour bar axes
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1)),
        ax=axes.ravel().tolist(),
    )  # This does a mapabble on the fly i think, not sure

    cbar.set_label("Weighting matrix")

    plotName = fileName + "/Prints"
    f = plotName + "/%s_prints_init_weighting_matrix.eps" % (property)
    fig.savefig(f, dpi=dpi_save, format="eps")


def prints_final_weighting_matrix(
    fileName: str,
    Data_list: list[Network],
    dpi_save: int,
    nrows: int,
    ncols: int,
    cmap,
    property_list: list,
    property,
    round_dec,
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):
        # print(i)
        mat = ax.matshow(
            Data_list[i].history_weighting_matrix[-1],
            cmap=cmap,
            norm=Normalize(vmin=0, vmax=1),
            aspect="auto",
        )
        ax.set_title(r"{} = {}".format(property, round(property_list[i], round_dec)))
        ax.set_xlabel("Agent Link Strength")
        ax.set_ylabel("Agent Link Strength")
    plt.tight_layout()

    # colour bar axes
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1)),
        ax=axes.ravel().tolist(),
    )  # This does a mapabble on the fly i think, not sure

    cbar.set_label("Weighting matrix")

    plotName = fileName + "/Prints"
    f = plotName + "/%s_prints_final_weighting_matrix.eps" % (property)
    fig.savefig(f, dpi=dpi_save, format="eps")


######################################################################################################################################################
"""Animations of changing culture"""


def live_compare_animate_culture_network_and_weighting(
    FILENAME: str,
    Data_list: list,
    layout: str,
    cmap_culture: Union[LinearSegmentedColormap, str],
    node_size: int,
    interval: int,
    fps: int,
    norm_zero_one: SymLogNorm,
    round_dec: int,
    cmap_edge,
    nrows,
    ncols,
    property_name,
    property_list,
):
    def update(i, Data_list, axes, cmap_culture, layout, title):

        for j, ax in enumerate(axes.flat):

            ax.clear()
            # print(Data["individual_culture"][i],Data["individual_culture"][i].shape)
            individual_culture_list = [x.culture for x in Data_list[j].agent_list]
            colour_adjust = norm_zero_one(individual_culture_list)
            ani_step_colours = cmap_culture(colour_adjust)

            G = nx.from_numpy_matrix(Data_list[j].history_weighting_matrix[i])

            # get pos
            pos = prod_pos(layout, G)

            weights = [G[u][v]["weight"] for u, v in G.edges()]
            norm = Normalize(vmin=0, vmax=1)
            colour_adjust_edge = norm(weights)
            colors_weights = cmap_edge(colour_adjust_edge)

            nx.draw(
                G,
                node_color=ani_step_colours,
                edge_color=colors_weights,
                ax=ax,
                pos=pos,
                node_size=node_size,
                edgecolors="black",
            )

            ax.set_title(
                r"%s = %s" % (property_name, round(property_list[j], round_dec))
            )

        title.set_text(
            "Time= {}".format(round(Data_list[0].history_time[i], round_dec))
        )

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    title = plt.suptitle(t="", fontsize=20)

    cbar_culture = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_culture),
        ax=axes.ravel().tolist(),
        location="right",
    )  # This does a mapabble on the fly i think, not sure
    cbar_culture.set_label("Identity")

    cbar_weight = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_edge),
        ax=axes.ravel().tolist(),
        location="left",
    )  # This does a mapabble on the fly i think, not sure
    cbar_weight.set_label("Link Strength")

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(len(Data_list[0].history_time)),
        fargs=(Data_list, axes, cmap_culture, layout, title),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = FILENAME + "/Animations"
    f = (
        animateName
        + "/live_multi_animate_culture_network_and_weighting_%s.mp4" % property_name
    )
    # print("f", f)
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)

    return ani


# animation of changing culture
def live_compare_animate_weighting_matrix(
    FILENAME: str,
    Data_list: list,
    cmap_weighting: Union[LinearSegmentedColormap, str],
    interval: int,
    fps: int,
    round_dec: int,
    cmap_edge,
    nrows,
    ncols,
    property_name,
    property_list,
):
    def update(i, Data_list, axes, title):

        for j, ax in enumerate(axes.flat):

            ax.clear()

            ax.matshow(
                Data_list[j].history_weighting_matrix[i],
                cmap=cmap_weighting,
                norm=Normalize(vmin=0, vmax=1),
                aspect="auto",
            )

            ax.set_title(
                r"%s = %s" % (property_name, round(property_list[j], round_dec))
            )
            ax.set_xlabel("Agent")
            ax.set_ylabel("Agent")

        title.set_text(
            "Time= {}".format(round(Data_list[0].history_time[i], round_dec))
        )

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(14, 7), constrained_layout=True
    )

    # plt.tight_layout()

    title = plt.suptitle(t="", fontsize=20)

    cbar_weight = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_weighting),
        ax=axes.ravel().tolist(),
        location="right",
    )  # This does a mapabble on the fly i think, not sure
    cbar_weight.set_label("Link Strength")

    # need to generate the network from the matrix
    # G = nx.from_numpy_matrix(Data_list[0].history_weighting_matrix[0])

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(len(Data_list[0].history_time)),
        fargs=(Data_list, axes, title),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/live_compare_animate_weighting_matrix_%s.mp4" % property_name
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)

    return ani


# animation of changing culture
def live_compare_animate_behaviour_matrix(
    FILENAME: str,
    Data_list: list,
    cmap_behaviour: Union[LinearSegmentedColormap, str],
    interval: int,
    fps: int,
    round_dec: int,
    nrows,
    ncols,
    property_name,
    property_list,
):
    def update(i, Data_list, axes, title):

        for j, ax in enumerate(axes.flat):

            ax.clear()

            for q in Data_list[j].agent_list:
                q.history_behaviour_values

            M = [n.history_behaviour_values[i] for n in Data_list[j].agent_list]

            ax.matshow(
                M,
                cmap=cmap_behaviour,
                norm=Normalize(vmin=-1, vmax=1),
                aspect="auto",
            )

            ax.set_title(
                r"%s = %s" % (property_name, round(property_list[j], round_dec))
            )
            ax.set_xlabel("Agent")
            ax.set_ylabel("Agent")

        title.set_text(
            "Time= {}".format(round(Data_list[0].history_time[i], round_dec))
        )

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(14, 7), constrained_layout=True
    )

    # fig.tight_layout(h_pad=2)
    # plt.tight_layout()

    title = plt.suptitle(t="", fontsize=20)

    cbar_weight = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_behaviour),
        ax=axes.ravel().tolist(),
        location="right",
    )  # This does a mapabble on the fly i think, not sure
    cbar_weight.set_label("Behavioural Value")

    # need to generate the network from the matrix
    # G = nx.from_numpy_matrix(Data_list[0].history_weighting_matrix[0])

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(len(Data_list[0].history_time)),
        fargs=(Data_list, axes, title),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/live_compare_animate_behaviour_matrix_%s.mp4" % property_name
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)

    return ani


def live_compare_plot_animate_behaviour_scatter(
    fileName,
    Data_list,
    norm_zero_one,
    cmap_culture,
    nrows,
    ncols,
    property_name,
    property_list,
    interval,
    fps,
    round_dec,
):
    def update(i, Data_list, axes, title):

        for j, ax in enumerate(axes.flat):
            ax.clear()

            individual_culture_list = [
                x.history_culture[i] for x in Data_list[j].agent_list
            ]  # where is the time step here?

            colour_adjust = norm_zero_one(individual_culture_list)
            ani_step_colours = cmap_culture(colour_adjust)

            x = [
                v.history_behaviour_attitudes[i][0] for v in Data_list[j].agent_list
            ]  # Data_list[j][property][i].T[0]
            y = [
                v.history_behaviour_attitudes[i][1] for v in Data_list[j].agent_list
            ]  # Data_list[j][property][i].T[1]

            # print(x,y)

            ax.scatter(x, y, s=60, c=ani_step_colours, edgecolors="black", linewidths=1)

            ax.set_xlabel(r"Attitude")
            ax.set_ylabel(r"Attitude")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(r"%s = %s" % (property_name, property_list[j]))

        title.set_text(
            "Time= {}".format(round(Data_list[0].history_time[i], round_dec))
        )

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(14, 7), constrained_layout=True
    )

    # plt.tight_layout()

    title = plt.suptitle(t="", fontsize=20)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_culture),
        ax=axes.ravel().tolist(),
        location="right",
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label("Identity")

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(len(Data_list[0].history_time)),
        fargs=(Data_list, axes, title),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = fileName + "/Animations"
    f = animateName + "/" + "live_compare_plot_animate_behaviour_scatter.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)


"""SA"""


def bar_sensitivity_analysis_plot(FILENAME, data, names, yerr, dpi_save, N_samples):
    """
    Create bar chart of results.
    """

    fig, ax = plt.subplots(figsize=(14, 7))
    data.plot(kind="barh", xerr=yerr, ax=ax)  # Pandas data frame plot
    ax.set_yticks(
        ticks=range(len(names)), labels=names
    )  # ,rotation=0.0, horizontalalignment="center"
    ax.set_xlim(left=0)
    plt.tight_layout()

    plotName = FILENAME + "/Prints"
    f = (
        plotName
        + "/"
        + "%s_%s_bar_sensitivity_analysis_plot.eps" % (len(names), N_samples)
    )
    fig.savefig(f, dpi=dpi_save, format="eps")


def scatter_total_sensitivity_analysis_plot(
    FILENAME, data, names, xerr, dpi_save, N_samples
):
    """
    Create scatter chart of results.
    """

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.errorbar(
        data["ST"].tolist(), names, xerr=xerr["ST"].tolist(), fmt="o", ecolor="k"
    )

    ax.set_xlim(left=0)
    ax.set_xlabel(r"Total Sobol sensitivity")
    ax.set_ylabel(r"Exogenous parameters")
    plt.tight_layout()

    plotName = FILENAME + "/Prints"
    f = (
        plotName
        + "/"
        + "%s_%s_scatter_sensitivity_analysis_plot.eps" % (len(names), N_samples)
    )
    fig.savefig(f, dpi=dpi_save, format="eps")


def multi_scatter_total_sensitivity_analysis_plot(
    FILENAME, data_dict, names, dpi_save, N_samples, order
):
    """
    Create scatter chart of results.
    """

    fig, ax = plt.subplots(figsize=(14, 7))

    if order == "First":
        for i in data_dict.values():
            l, caps, c = ax.errorbar(
                i["data"]["S1"].tolist(),
                names,
                xerr=i["yerr"]["S1"].tolist(),
                fmt="o",
                ecolor="k",
                color=i["colour"],
                label=i["title"],
            )
    elif order == "Total":
        for i in data_dict.values():
            l, caps, c = ax.errorbar(
                i["data"]["ST"].tolist(),
                names,
                xerr=i["yerr"]["ST"].tolist(),
                fmt="o",
                ecolor="k",
                color=i["colour"],
                label=i["title"],
            )
    else:
        print("INVALID ORDER")

    ax.legend()
    ax.set_xlim(left=0)
    ax.set_xlabel(r"%s Sobol sensitivity" % (order))
    ax.set_ylabel(r"Exogenous parameters")
    plt.tight_layout()

    plotName = FILENAME + "/Prints"
    f = (
        plotName
        + "/"
        + "%s_%s_%s_multi_scatter_sensitivity_analysis_plot.eps"
        % (len(names), N_samples, order)
    )
    fig.savefig(f, dpi=dpi_save, format="eps")


def multi_scatter_seperate_total_sensitivity_analysis_plot(
    FILENAME, data_dict, names, dpi_save, N_samples, order
):
    """
    Create scatter chart of results.
    """

    fig, axes = plt.subplots(
        figsize=(14, 7), ncols=len(list(data_dict.keys())), nrows=1
    )
    dict_list = list(data_dict.keys())

    for i, ax in enumerate(axes.flat):
        ax.errorbar(
            data_dict[dict_list[i]]["data"]["S1"].tolist(),
            names,
            xerr=data_dict[dict_list[i]]["yerr"]["S1"].tolist(),
            fmt="o",
            ecolor="k",
            color=data_dict[dict_list[i]]["colour"],
            label=data_dict[dict_list[i]]["title"],
        )
        ax.legend()
        ax.set_xlim(left=0)
        ax.set_xlabel(r"%s Sobol sensitivity" % (order))
        ax.set_ylabel(r"Exogenous parameters")

    plt.tight_layout()

    plotName = FILENAME + "/Prints"
    f = (
        plotName
        + "/"
        + "%s_%s_%s_multi_scatter_seperate_sensitivity_analysis_plot.eps"
        % (len(names), N_samples, order)
    )
    fig.savefig(f, dpi=dpi_save, format="eps")


def multi_scatter_sidebyside_total_sensitivity_analysis_plot(
    FILENAME, data_dict, names, dpi_save, N_samples, order
):
    """
    Create scatter chart of results.
    """

    fig, ax = plt.subplots(figsize=(14, 7))

    padding = 0.15
    width_max = 1
    width_scatter = (width_max - 2 * padding) / len(
        list(data_dict.values())
    )  # width over which scatter points are placed
    y_height_list = np.asarray(list(range(len(names)))) + padding
    y_height_list_copy = np.asarray(list(range(len(names)))) + padding

    offset = 0
    if order == "First":
        for i in data_dict.values():
            y_height_list_copy = y_height_list_copy + offset
            print("y_height_list", y_height_list_copy)
            ax.errorbar(
                i["data"]["S1"].tolist(),
                y_height_list_copy,
                xerr=i["yerr"]["S1"].tolist(),
                fmt="o",
                ecolor="k",
                color=i["colour"],
                label=i["title"],
            )
            offset += width_scatter
    elif order == "Total":
        for i in data_dict.values():
            y_height_list_copy = y_height_list_copy + offset
            ax.errorbar(
                i["data"]["ST"].tolist(),
                y_height_list_copy,
                xerr=i["yerr"]["ST"].tolist(),
                fmt="o",
                ecolor="k",
                color=i["colour"],
                label=i["title"],
            )
            offset += width_scatter
    else:
        print("INVALID ORDER")

    ax.hlines(
        y=np.asarray(list(range(len(names) - 1))) + 1,
        xmin=0,
        xmax=1,
        linestyles="dashed",
        colors="k",
        alpha=0.3,
    )
    ax.set_yticks(np.asarray(list(range(len(names)))) + width_max / 2, names)

    ax.legend()
    ax.set_ylim(bottom=0, top=len(names))
    ax.set_xlim(left=0, right=1)
    ax.set_xlabel(r"%s Sobol sensitivity" % (order))
    ax.set_ylabel(r"Exogenous parameters")
    plt.tight_layout()

    plotName = FILENAME + "/Prints"
    f = (
        plotName
        + "/"
        + "%s_%s_%s_multi_scatter_sidebyside_total_sensitivity_analysis_plot.eps"
        % (len(names), N_samples, order)
    )
    fig.savefig(f, dpi=dpi_save, format="eps")


def prints_SA_matrix(
    FILENAME, Data, title_list, cmap, nrows, ncols, dpi_save, labels, N_samples, title_property, Y_property
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    type_matrix = ["index","confidence"]
    for i, ax in enumerate(axes.flat):
        matrix = ax.matshow(
            Data[i],
            cmap=cmap,
            aspect="auto",
        )
        #ax.set_title(title_list[i])
        # colour bar axes
        cbar = fig.colorbar(
            matrix, ax=ax, label = "Second order sobol %s: %s" % (type_matrix[i],title_property)
        )  # This does a mapabble on the fly i think, not sure
        xaxis = np.arange(len(labels))
        ax.set_xticks(xaxis)
        ax.set_yticks(xaxis)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels, rotation=0)
        # ax.xticks(rotation=45, ha='right')
    # plt.tight_layout()

    plotName = FILENAME + "/Prints"
    f = plotName + "/" + "%s_%s_prints_SA_matrix_property_%s.eps" % (len(labels), N_samples,Y_property)
    fig.savefig(f, dpi=dpi_save, format="eps")


#############################################################################################################################################

"""MISC"""


def live_print_culture_histgram(
    fileName: str,
    Data_list: list[Network],
    title_list: str,
    nrows: int,
    ncols: int,
    dpi_save: int,
):
    y_title = "indivdiual culture"

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):

        ax.set_title(title_list[i])
        ax.set_ylabel(r"%s" % y_title)

        ax.hist(x, density=True, bins=30)  # density=False would make counts
        for v in Data_list[i].agent_list:
            ax.plot(
                np.asarray(Data_list[i].history_time), np.asarray(v.history_culture)
            )
        ax.set_xlabel(r"Time")

    plotName = fileName + "/Prints"
    f = plotName + "/print_culture_timeseries.eps"
    fig.savefig(f, dpi=dpi_save, format="eps")


def live_plot_attitude_scatter(fileName, Data, dpi_save):
    attitudes_list = []

    for i in Data.M:
        attitudes = np.asarray(
            [[v.history_attitudes for v in i] for i in Data.agent_list]
        )
        attitudes_list.append(attitudes.T)

    fig, ax = plt.subplots()
    ax.scatter(attitudes_list[0][-1], attitudes_list[1][-1])
    ax.set_xlabel(r"Attitude")
    ax.set_ylabel(r"Attitude")

    plotName = fileName + "/Plots"
    f = plotName + "/live_plot_attitude_scatter.eps"
    fig.savefig(f, dpi=dpi_save, format="eps")


def plot_alpha_variation(FILENAME, num_counts, phi_list, dpi_save):
    def alpha_calc(phi, x):
        return np.exp(-phi * np.abs(x))

    def alpha_diff_calc(phi, x):
        return -phi * np.exp(-phi * np.abs(x))

    fig, ax = plt.subplots()

    x = np.linspace(0, 1, num_counts)

    for i in phi_list:
        y = [alpha_calc(i, x) for x in x]
        ax.plot(x, y, "-", label="Phi = %s" % i)
        dydx = [alpha_diff_calc(i, x) for x in x]
        ax.plot(x, dydx, "--", label="Phi = %s" % i)

    ax.set_xlabel(r"$|I_n -I_k|$")
    ax.set_ylabel(r"$\alpha$")
    ax.legend()
    plotName = FILENAME + "/Plots"
    f = plotName + "/plot_alpha_variation.eps"
    fig.savefig(f, dpi=dpi_save, format="eps")


def prod_pos(layout_type: str, network: Graph) -> Graph:

    if layout_type == "circular":
        pos_culture_network = nx.circular_layout(network)
    elif layout_type == "spring":
        pos_culture_network = nx.spring_layout(network)
    elif layout_type == "kamada_kawai":
        pos_culture_network = nx.kamada_kawai_layout(network)
    elif layout_type == "planar":
        pos_culture_network = nx.planar_layout(network)
    else:
        raise Exception("Invalid layout given")

    return pos_culture_network


def frame_distribution(
    time_list: list, scale_factor: int, frames_proportion: float
) -> list:
    # print("SEKEET",type(scale_factor))
    select = np.random.exponential(scale=scale_factor, size=frames_proportion)
    # print(select)
    norm_select = select / max(select)
    # print(norm_select)
    scaled_select = np.round(norm_select * (len(time_list) - 1))
    # print(scaled_select)
    frames_list = np.unique(scaled_select)
    # print(frames_list)
    frames_list_int = [int(x) for x in frames_list]
    print("frames:", frames_list_int)
    return frames_list_int


############################################################################################################################

# SAVED DATA PLOTS - NEED TO CONVERT
"""
def plot_behaviour_scatter(fileName,Data,property,dpi_save):
    PropertyData = Data[property].transpose()
    
    fig, ax = plt.subplots()

    for j in range(int(Data["N"])):
        ax.scatter(PropertyData[0][j][-1], PropertyData[1][j][-1])

    ax.set_xlabel(r"Attitude")
    ax.set_ylabel(r"Attitude")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_attitude_scatter.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')

def plot_weighting_link_timeseries(FILENAME: str, Data: DataFrame, y_title:str, dpi_save:int, min_val):

    fig, ax = plt.subplots()

    for i in range(int(Data["N"])):
        for v in range(int(Data["N"])):
            if Data["network_weighting_matrix"][0][i][v] > 0.0:
                link_data = [Data["network_weighting_matrix"][x][i][v] for x in range(len(Data["network_time"]))]
                if any(j > min_val for j in link_data):
                    ax.plot(Data["network_time"], link_data)

    ax.axvline(Data["culture_momentum_real"], color='r',linestyle = "--")
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    plotName = FILENAME + "/Plots"
    f = plotName + "/plot_weighting_link_timeseries.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')

def animate_culture_network_and_weighting(
    FILENAME: str, Data: DataFrame, layout:str, cmap_culture: Union[LinearSegmentedColormap,str], node_size:int, interval:int, fps:int, norm_zero_one: SymLogNorm, round_dec:int, cmap_edge
):

    def update(i, G, pos, ax, cmap_culture):

        ax.clear()
        # print(Data["individual_culture"][i],Data["individual_culture"][i].shape)
        
        colour_adjust = norm_zero_one(Data["individual_culture"][i])
        ani_step_colours = cmap_culture(colour_adjust)

        G = nx.from_numpy_matrix( Data["network_weighting_matrix"][i])

        weights = [G[u][v]['weight'] for u,v in G.edges()]
        norm = Normalize(vmin=0, vmax=1)
        colour_adjust_edge = norm(weights)
        colors_weights = cmap_edge(colour_adjust_edge)

        nx.draw(
            G,
            node_color=ani_step_colours,
            edge_color=colors_weights,
            ax=ax,
            pos=pos,
            node_size=node_size,
            edgecolors="black",
        )

        # Set the title
        ax.set_title("Time= {}".format(round(Data["network_time"][i], round_dec)))

    # Build plot
    fig, ax = plt.subplots()
    # cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax)#This does a mapabble on the fly i think, not sure
    cbar_culture = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax
    )  # This does a mapabble on the fly i think, not sure
    cbar_culture.set_label("Identity")

    cbar_weight = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_edge), ax=ax, location='left',
    )  # This does a mapabble on the fly i think, not sure
    cbar_weight.set_label("Link Strength")

    # need to generate the network from the matrix
    G = nx.from_numpy_matrix(Data["network_weighting_matrix"][0])

    # get pos
    pos_culture_network = prod_pos(layout, G)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(len(Data["network_time"])),
        fargs=(G, pos_culture_network, ax, cmap_culture),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "animate_culture_network_and_weighting.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)

    return ani

def animate_behaviour_scatter(fileName,Data,property,norm_zero_one, cmap_culture,interval, fps,round_dec):
    
    def update(i):
        ax.clear()

        colour_adjust = norm_zero_one(Data["individual_culture"][i])
        ani_step_colours = cmap_culture(colour_adjust)

        ax.scatter( Data[property][i].T[0], Data[property][i].T[1], s= 60, c = ani_step_colours, edgecolors='black', linewidths=1 )

        ax.set_xlabel(r"Attitude")
        ax.set_ylabel(r"Attitude")
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_title("Time= {}".format(round(Data["network_time"][i], round_dec)))

    fig, ax = plt.subplots()
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label("Identity")

    #print(Data[property][0].T,Data[property][0].T.shape )
    ax.scatter(Data[property][0].T[0], Data[property][0].T[1], s= 60)

    ax.set_xlabel(r"Attitude")
    ax.set_ylabel(r"Attitude")

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(len(Data["network_time"])),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = fileName + "/Animations"
    f = animateName + "/" + "attitude_scatter_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)

def print_culture_histogram(
    FILENAME: str, Data: DataFrame, property:str, nrows:int, ncols:int, frames_list, round_dec, dpi_save,bin_num
):
    y_title = "Probability"
    #print(Data[property], Data[property].shape)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    # print("property = ", property)

    for i, ax in enumerate(axes.flat):
        #print(Data[property][frames_list[i]])
        ax.hist(Data[property][frames_list[i]], density=True, bins = bin_num)  # density=False would make counts
        ax.set_xlabel(r"Identity")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title("Time= {}".format(round(Data["network_time"][frames_list[i]], round_dec)))  # avoid 0 in the title
    plt.tight_layout()

    plotName = FILENAME + "/Plots"
    f = plotName + "/print_culture_histogram.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')

    # make matrix animation
def animate_weighting_matrix(FILENAME: str, Data: DataFrame, interval:int, fps:int, round_dec:int, cmap_weighting: Union[LinearSegmentedColormap,str]):
    def update(i):
        M = Data["network_weighting_matrix"][i]
        # print("next frame!",M)        
        matrice.set_array(M)
        # Set the title
        
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][i], round_dec))
            )
        return matrice

    fig, ax = plt.subplots()
    ax.set_xlabel("Agent")
    ax.set_ylabel("Agent")
    matrice = ax.matshow(Data["network_weighting_matrix"][0], cmap=cmap_weighting)
    plt.colorbar(matrice)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(len(Data["network_time"])),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "weighting_matrix_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)
    return ani

# make behaviour evolution plot
def animate_behavioural_matrix(
    FILENAME: str, Data: DataFrame, interval:int, fps:int, cmap_behaviour: Union[LinearSegmentedColormap,str], round_dec:int
):
    def update(i):
        M = Data["behaviour_value"][i]
        # print("next frame!",M)
        matrice.set_array(M)

        # Set the title
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][i], round_dec))
            )

        return matrice

    fig, ax = plt.subplots()
    ax.set_xlabel("Behaviour")
    ax.set_ylabel("Agent")

    matrice = ax.matshow(Data["behaviour_value"][0], cmap=cmap_behaviour, aspect="auto")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=-1, vmax=1)),
        ax=ax,
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label("Behavioural Value")

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(len(Data["network_time"])),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "behavioural_matrix_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)
    return ani

# animation of changing culture
def animate_culture_network(
    FILENAME: str, Data: DataFrame, layout:str, cmap_culture: Union[LinearSegmentedColormap,str], node_size:int, interval:int, fps:int, norm_zero_one: SymLogNorm, round_dec:int
):
    def update(i, G, pos, ax, cmap_culture):

        ax.clear()
        # print(Data["individual_culture"][i],Data["individual_culture"][i].shape)
        colour_adjust = norm_zero_one(Data["individual_culture"][i])
        ani_step_colours = cmap_culture(colour_adjust)
        nx.draw(
            G,
            node_color=ani_step_colours,
            ax=ax,
            pos=pos,
            node_size=node_size,
            edgecolors="black",
        )

        # Set the title
        ax.set_title("Time= {}".format(round(Data["network_time"][i], round_dec)))

    # Build plot
    fig, ax = plt.subplots()
    # cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax)#This does a mapabble on the fly i think, not sure
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label("Identity")

    # need to generate the network from the matrix
    G = nx.from_numpy_matrix(Data["network_weighting_matrix"][0])

    # get pos
    pos_culture_network = prod_pos(layout, G)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(len(Data["network_time"])),
        fargs=(G, pos_culture_network, ax, cmap_culture),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "cultural_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)
    return ani

def prints_weighting_matrix(
    FILENAME: str, Data: DataFrame, cmap_behaviour: Union[LinearSegmentedColormap,str], nrows:int, ncols:int, frames_list:list[int], round_dec:int, dpi_save:int
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    # print(frames_list,Data["network_time"],Data["network_weighting_matrix"][20])
    #print( len(Data["network_weighting_matrix"]),frames_list)
    for i, ax in enumerate(axes.flat):
        # print(i)
        ax.matshow(
            Data["network_weighting_matrix"][frames_list[i]],
            cmap=cmap_behaviour,
            norm=Normalize(vmin=0, vmax=1),
            aspect="auto",
        )
        # Set the title
        # print("Time= {}".format(round(Data["network_time"][frames_list[i]],round_dec)))
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][frames_list[i]], round_dec))
        )
        ax.set_xlabel("Agent Link Strength")
        ax.set_ylabel("Agent Link Strength")
    plt.tight_layout()

    # colour bar axes
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=0, vmax=1)),ax=axes.ravel().tolist())  # This does a mapabble on the fly i think, not sure
    cbar.set_label("Weighting matrix")

    plotName = FILENAME + "/Prints"
    f = plotName + "/" + "prints_weighting_matrix.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')


def prints_behavioural_matrix(
    FILENAME: str, Data: DataFrame, cmap_behaviour: Union[LinearSegmentedColormap,str], nrows:int, ncols:int, frames_list:list[int], round_dec:int, dpi_save:int
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):

        ax.matshow(
            Data["behaviour_value"][frames_list[i]], cmap=cmap_behaviour, aspect="auto"
        )
        # Set the title
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][frames_list[i]], round_dec))
        )
        ax.set_xlabel("Behaviour")
        ax.set_ylabel("Agent")
    plt.tight_layout()

    # colour bar axes
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=0, vmax=1)),ax=axes.ravel().tolist())
    cbar.set_label("Behavioural Value")

    plotName = FILENAME + "/Prints"
    f = plotName + "/" + "prints_behavioural_matrix.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')

def print_network_social_component_matrix(
    FILENAME: str, Data: DataFrame, cmap_behaviour: Union[LinearSegmentedColormap,str], nrows:int, ncols:int, frames_list:list[int], round_dec:int, dpi_save:int
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    # print(frames_list,Data["network_time"],Data["network_weighting_matrix"][20])
    #print( len(Data[ "network_social_component_matrix"]),frames_list)
    for i, ax in enumerate(axes.flat):
        # print(i)
        ax.matshow(
            Data["network_social_component_matrix"][frames_list[i]],
            cmap=cmap_behaviour,
            aspect="auto",
        )
        # Set the title
        # print("Time= {}".format(round(Data["network_time"][frames_list[i]],round_dec)))
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][frames_list[i]], round_dec))
        )
        ax.set_xlabel("Behaviour")
        ax.set_ylabel("Agent")
    plt.tight_layout()

    # colour bar axes
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=-1, vmax=1)),ax=axes.ravel().tolist())
    cbar.set_label("Social Learning")

    plotName = FILENAME + "/Prints"
    f = plotName + "/" + "prints_network_social_component_matrix.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')

def print_network_information_provision(
    FILENAME: str, Data: DataFrame, cmap_behaviour: Union[LinearSegmentedColormap,str], nrows:int, ncols:int, frames_list:list[int], round_dec:int, dpi_save:int
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    # print(frames_list,Data["network_time"],Data["network_weighting_matrix"][20])
    #print( len(Data[ "network_social_component_matrix"]),frames_list)
    #
    #print(Data["behaviour_information_provision"].max())
    
    for i, ax in enumerate(axes.flat):
        # print(i)
        ax.matshow(
            Data["behaviour_information_provision"][frames_list[i]],
            cmap=cmap_behaviour,
            aspect="auto",
        )
        # Set the title
        # print("Time= {}".format(round(Data["network_time"][frames_list[i]],round_dec)))
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][frames_list[i]], round_dec))
        )
        ax.set_xlabel("Behaviour")
        ax.set_ylabel("Agent")
    plt.tight_layout()

    # colour bar axes
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=0, vmax=1)),ax=axes.ravel().tolist())
    cbar.set_label("Information Provision")

    plotName = FILENAME + "/Prints"
    f = plotName + "/" + "prints_behaviour_information_provision_matrix.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')


def prints_culture_network(
    FILENAME: str, Data: DataFrame,layout:str, cmap_culture: LinearSegmentedColormap,node_size:int, nrows:int, ncols:int, norm_zero_one: SymLogNorm, frames_list:list[int], round_dec:int, dpi_save:int
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    # need to generate the network from the matrix
    G = nx.from_numpy_matrix(Data["network_weighting_matrix"][0])

    # get pos
    pos_culture_network = prod_pos(layout, G)

    for i, ax in enumerate(axes.flat):
        # print(i,ax)
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][frames_list[i]], round_dec))
        )

        colour_adjust = norm_zero_one(Data["individual_culture"][frames_list[i]])
        #colour_adjust = (Data["individual_culture"][frames_list[i]] + 1)/2
        #colour_adjust = Data["individual_culture"][frames_list[i]]
        #print("colour_adjust", colour_adjust)
        ani_step_colours = cmap_culture(colour_adjust)
        #print("ani_step_colours",ani_step_colours)

        nx.draw(
            G,
            node_color=ani_step_colours,
            ax=ax,
            pos=pos_culture_network,
            node_size=node_size,
            edgecolors="black",
        )

    plt.tight_layout()

    # colour bar axes
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture, norm=norm_zero_one),ax=axes.ravel().tolist())
    cbar.set_label("Identity")

    f = FILENAME + "/Prints/prints_culture_network.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')

def plot_average_culture_timeseries(FILENAME: str, Data: DataFrame, dpi_save:int):
    y_title = "Average Culture"
    property = "network_average_culture"

    fig, ax = plt.subplots()
    data = np.asarray(Data[property])[0]  # bodge
    culture_min = np.asarray(Data["network_min_culture"])[0]  # bodge
    culture_max = np.asarray(Data["network_max_culture"])[0]  # bodge
    ax.plot(Data["network_time"], data)
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)
    ax.axvline(Data["culture_momentum_real"], color='r',linestyle = "--")
    ax.fill_between(
        Data["network_time"], culture_min, culture_max, alpha=0.5, linewidth=0
    )

    plotName = FILENAME + "/Plots"
    f = plotName + "/" + property + "_timeseries.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')


def plot_culture_timeseries(FILENAME: str, Data: DataFrame, dpi_save:int):

    ##plot cultural evolution of agents
    fig, ax = plt.subplots()
    ax.set_xlabel("Time")
    ax.set_ylabel("Identity")

    data = np.asarray(Data["individual_culture"])  # bodge

    for i in range(int(int(Data["N"]))):
        # print(Data["individual_culture"][i])
        ax.plot(Data["network_time"], data[i])
    ax.axvline(Data["culture_momentum_real"], color='r',linestyle = "--")

    plotName = FILENAME + "/Plots"
    f = plotName + "/" + "cultural_evolution.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')

# make animate_network_social_component_matrix
def animate_network_social_component_matrix(FILENAME: str, Data: DataFrame, interval:int, fps:int, round_dec:int, cmap: Union[LinearSegmentedColormap,str], norm_zero_one):
    
    def update(i):
        M = Data["network_social_component_matrix"][i]
        # print("next frame!",M)        
        matrice.set_array(M)
        # Set the title        
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][i], round_dec))
            )
        return matrice

    fig, ax = plt.subplots()

    matrice = ax.matshow(Data["network_social_component_matrix"][0], cmap=cmap, aspect="auto")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Behaviour")
    #plt.colorbar(matrice)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm_zero_one),
        ax=ax,
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label("Social Learning")

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(len(Data["network_time"])),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "network_social_component_matrix_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)

    return ani

def animate_network_information_provision(FILENAME: str, Data: DataFrame, interval:int, fps:int, round_dec:int, cmap: Union[LinearSegmentedColormap,str]):

    def update(i):
        M = Data["behaviour_information_provision"][i]
        #print("next frame!",M, np.shape(M))  
      
        matrice.set_array(M)
        # Set the title        
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][i], round_dec))
            )
        return matrice

    fig, ax = plt.subplots()

    matrice = ax.matshow(Data["behaviour_information_provision"][0], cmap=cmap, aspect="auto")
    ax.set_xlabel("Behaviour")
    ax.set_ylabel("Agent")
    #plt.colorbar(matrice)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=Data["behaviour_information_provision"].max())),
        ax=ax,
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label("Information Provision")

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(len(Data["network_time"])),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "behaviour_information_provision_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)
    return ani

def prints_behaviour_timeseries_plot(FILENAME: str, Data: DataFrame, property:str, y_title:str, nrows:int, ncols:int, dpi_save:int):
    PropertyData = Data[property].transpose()

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    # print("property = ", property)

    for i, ax in enumerate(axes.flat):
        for j in range(int(Data["N"])):
            ax.plot(Data["network_time"], PropertyData[i][j])
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title(r'$\phi$ = ' + str(Data["phi_list"][i]))  # avoid 0 in the title
        ax.axvline(Data["culture_momentum_real"], color='r',linestyle = "--")
        
    plt.tight_layout()
    
    plotName = FILENAME + "/Plots"
    f = plotName + "/" + property + "_prints_timeseries.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')

def prints_behaviour_timeseries_plot_colour_culture(
    FILENAME: str, Data: DataFrame, property:str, y_title:str, nrows:int, ncols:int, dpi_save:int, culture_cmap, norm_zero_one
):

    PropertyData = Data[property].T
    culture_n_T = Data["individual_culture"]
    bodge_culture = np.delete(np.asarray(culture_n_T), -1, axis=1)#NEEDS TO MATCH THE LENGTH OF THE SEGMENTS REMOVE THE END AS IT WILL BE THE LEAST LIKELY TO CHANGE FROM ONE STEP TO ANOTHER

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7), constrained_layout=True)
    for i, ax in enumerate(axes.flat):
        for j in range(int(Data["N"])):
            # TAKEN FROM THE MATPLOTLIB EXAMPLE
            # Create a set of line segments so that we can color them individually
            # This creates the points as a N x 1 x 2 array so that we can stack points
            # together easily to get the segments. The segments array for line collection
            # needs to be (numlines) x (points per line) x 2 (for x and y)
            points = np.array([Data["network_time"],PropertyData[i][j]]).T.reshape(-1, 1, 2)# Reshapes from (2, t) to (t,2) to (t, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)# Reshapes from (t, 1, 2) to (t-1, 2, 2)

            # Create a continuous norm to map from data points to colors
            lc = LineCollection(segments, cmap = culture_cmap, norm=norm_zero_one)
            lc.set_array(bodge_culture[j])#add the colours to the lines
            ax.add_collection(lc)#add the line to the axis

        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_xlim(0,Data["network_time"][-1])# just in case
        ax.set_title(r'$\phi$ = ' + str(Data["phi_list"][i]))  # avoid 0 in the title

    cbar_culture = fig.colorbar(
        plt.cm.ScalarMappable(cmap=culture_cmap), ax=axes.ravel().tolist(), location='right',
    )
    cbar_culture.set_label(r"Identity")

    plotName = FILENAME + "/Plots"
    f = plotName + "/" + property + "_prints_behaviour_timeseries_plot_colour_culture.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')


def standard_behaviour_timeseries_plot(FILENAME: str, Data: DataFrame, property:str, y_title:str, dpi_save:int):
    PropertyData = Data[property].transpose()

    fig, ax = plt.subplots()
    for i in range(int(Data["N"])):
        for v in range(int(Data["M"])):
            ax.plot(Data["network_time"], PropertyData[i][v])
    ax.axvline(Data["culture_momentum_real"], color='r',linestyle = "--")
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    plotName = FILENAME + "/Plots"
    f = plotName + "/" + property + "_timeseries.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')


def plot_value_timeseries(FILENAME: str, Data: DataFrame, nrows:int, ncols:int, dpi_save:int,):
    prints_behaviour_timeseries_plot(
        FILENAME, Data, "behaviour_value", "Trait Value", nrows, ncols, dpi_save, 
    )


def plot_threshold_timeseries(FILENAME: str, Data: DataFrame, nrows:int, ncols:int, dpi_save:int,):
    prints_behaviour_timeseries_plot(
        FILENAME, Data, "behaviour_threshold", "Threshold", nrows, ncols, dpi_save
    )


def plot_attitude_timeseries(FILENAME: str, Data: DataFrame, nrows:int, ncols:int, dpi_save:int):
    
    #print(Data["behaviour_attitude"],np.shape(Data["behaviour_attitude"]))


    prints_behaviour_timeseries_plot(
        FILENAME, Data, "behaviour_attitude", "Attitude", nrows, ncols, dpi_save,
    )


def plot_av_carbon_emissions_timeseries(FILENAME: str, Data: DataFrame, dpi_save:int):
    y_title = "Carbon Emissions Per Individual"
    property = "individual_carbon_emissions"

    fig, ax = plt.subplots()
    av_network_total_carbon_emissions = [
        x / Data["N"] for x in np.asarray(Data["network_total_carbon_emissions"])[0]
    ]
    ax.plot(Data["network_time"], av_network_total_carbon_emissions, "k-")
    ax.axvline(Data["culture_momentum_real"], color='r',linestyle = "--")
    data = np.asarray(Data["individual_carbon_emissions"])  # bodge

    for i in range(int(int(Data["N"]))):
        ax.plot(Data["network_time"], data[i])

    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    plotName = FILENAME + "/Plots"
    f = plotName + "/" + property + "_timeseries.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')


"""
