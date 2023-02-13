"""Plot simulation results
A module that use input data or social network object to produce plots for analysis.
These plots also include animations or phase diagrams.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import string
from matplotlib.lines import lineStyles
import networkx as nx
from networkx import Graph
import numpy as np
#from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize, LinearSegmentedColormap, SymLogNorm
from matplotlib.image import NonUniformImage
from matplotlib.collections import LineCollection
from matplotlib.cm import get_cmap
from typing import Union
#from pydlc import dense_lines
from resources.network import Network
#import joypy
#from resources.utility import calc_num_clusters_auto_bandwidth
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity
from sklearn.cluster import MeanShift
#import pandas as pd
from resources.utility import get_cluster_list


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

def plot_four(fileName, emissions_id_array_no_green_no_culture, emissions_id_array_no_green_culture, emissions_id_array_green_no_culture, emissions_id_array_green_culture, confirmation_bias,mean_list, dpi_save):

    fig, ax = plt.subplots(figsize=(10,7)) 

    print("emissions_id_array_no_green_no_culture ",emissions_id_array_no_green_no_culture, emissions_id_array_no_green_no_culture.shape)
    mu_emissions_id_array_no_green_no_culture = emissions_id_array_no_green_no_culture.mean(axis=1)
    min_emissions_id_array_no_green_no_culture = emissions_id_array_no_green_no_culture.min(axis=1)
    max_emissions_id_array_no_green_no_culture = emissions_id_array_no_green_no_culture.max(axis=1)

    mu_emissions_id_array_no_green_culture = emissions_id_array_no_green_culture.mean(axis=1)
    min_emissions_id_array_no_green_culture = emissions_id_array_no_green_culture.min(axis=1)
    max_emissions_id_array_no_green_culture = emissions_id_array_no_green_culture.max(axis=1)

    mu_emissions_id_array_green_no_culture = emissions_id_array_green_no_culture.mean(axis=1)
    min_emissions_id_array_green_no_culture = emissions_id_array_green_no_culture.min(axis=1)
    max_emissions_id_array_green_no_culture = emissions_id_array_green_no_culture.max(axis=1)

    mu_emissions_id_array_green_culture = emissions_id_array_green_culture.mean(axis=1)
    min_emissions_id_array_green_culture = emissions_id_array_green_culture.min(axis=1)
    max_emissions_id_array_green_culture = emissions_id_array_green_culture.max(axis=1)


    # cultuer vs no culteur repsresneted black vs red
    # green vs no green by solid vs dashed line
    ax.plot(mean_list[::-1],mu_emissions_id_array_no_green_no_culture, linestyle="-", color='red', label = r"Behavioural independence, No green influencers")
    ax.fill_between(mean_list[::-1], min_emissions_id_array_no_green_no_culture, max_emissions_id_array_no_green_no_culture, facecolor='red', alpha=0.5)

    ax.plot(mean_list[::-1],mu_emissions_id_array_no_green_culture, linestyle="-", color='black', label = r"Behavioural dependence, No green influencers")
    ax.fill_between(mean_list[::-1], min_emissions_id_array_no_green_culture, max_emissions_id_array_no_green_culture, facecolor='black', alpha=0.5)

    ax.plot(mean_list[::-1],mu_emissions_id_array_green_no_culture, linestyle="--", color='red', label = r"Behavioural independence, Green influencers")
    ax.fill_between(mean_list[::-1], min_emissions_id_array_green_no_culture, max_emissions_id_array_green_no_culture, facecolor='red', alpha=0.5)

    ax.plot(mean_list[::-1],mu_emissions_id_array_green_culture, linestyle="-", color='black', label = r"Behavioural dependence, Green influencers")
    ax.fill_between(mean_list[::-1], min_emissions_id_array_green_culture, max_emissions_id_array_green_culture, facecolor='black', alpha=0.5)

    ax.legend()
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_four_%s" % (len(mean_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")




def plot_four_compare(fileName, emissions_difference_matrix_compare_green, emissions_difference_matrix_compare_no_green, emissions_difference_matrix_compare_culture, emissions_difference_matrix_compare_no_culture, theta, mean_list, dpi_save):
    fig, axes = plt.subplots(nrows = 2, ncols = 2,figsize=(10,7))    

    mu_emissions_difference_matrix_compare_green = emissions_difference_matrix_compare_green.mean(axis=1)
    min_emissions_difference_matrix_compare_green = emissions_difference_matrix_compare_green.min(axis=1)
    max_emissions_difference_matrix_compare_green = emissions_difference_matrix_compare_green.max(axis=1)

    mu_emissions_difference_matrix_compare_no_green = emissions_difference_matrix_compare_no_green.mean(axis=1)
    min_emissions_difference_matrix_compare_no_green = emissions_difference_matrix_compare_no_green.min(axis=1)
    max_emissions_difference_matrix_compare_no_green = emissions_difference_matrix_compare_no_green.max(axis=1)

    mu_emissions_difference_matrix_compare_culture = emissions_difference_matrix_compare_culture.mean(axis=1)
    min_emissions_difference_matrix_compare_culture = emissions_difference_matrix_compare_culture.min(axis=1)
    max_emissions_difference_matrix_compare_culture = emissions_difference_matrix_compare_culture.max(axis=1)

    mu_emissions_difference_matrix_compare_no_culture = emissions_difference_matrix_compare_no_culture.mean(axis=1)
    min_emissions_difference_matrix_compare_no_culture = emissions_difference_matrix_compare_no_culture.min(axis=1)
    max_emissions_difference_matrix_compare_no_culture = emissions_difference_matrix_compare_no_culture.max(axis=1)
    
    axes[0][0].plot(mean_list[::-1],mu_emissions_difference_matrix_compare_green, ls="", marker=".", linewidth = 0.5, color='blue', label = r"Confirmation bias $\theta = %s$"% (theta))
    axes[0][0].fill_between(mean_list[::-1], min_emissions_difference_matrix_compare_green, max_emissions_difference_matrix_compare_green, facecolor='blue', alpha=0.5)
    axes[0][0].set_title("CULTURE VS NO CULTURE WITH GREEN")
    axes[0][0].set_xlabel(r"Initial attitude distance, $1-a_A/(a_A + b_A)$")
    axes[0][0].set_ylabel( r"Relative $\%$ change in final emissions")
    
    axes[0][1].plot(mean_list[::-1],mu_emissions_difference_matrix_compare_no_green, ls="", marker=".", linewidth = 0.5, color='blue', label = r"Confirmation bias $\theta = %s$"% (theta))
    axes[0][1].fill_between(mean_list[::-1], min_emissions_difference_matrix_compare_no_green, max_emissions_difference_matrix_compare_no_green, facecolor='blue', alpha=0.5)
    axes[0][1].set_title("CULTURE VS NO CULTURE WITH  NO GREEN")
    axes[0][1].set_xlabel(r"Initial attitude distance, $1-a_A/(a_A + b_A)$")
    axes[0][1].set_ylabel( r"Relative $\%$ change in final emissions")

    axes[1][0].plot(mean_list[::-1],mu_emissions_difference_matrix_compare_culture, ls="", marker=".", linewidth = 0.5, color='blue', label = r"Confirmation bias $\theta = %s$"% (theta))
    axes[1][0].fill_between(mean_list[::-1], min_emissions_difference_matrix_compare_culture, max_emissions_difference_matrix_compare_culture, facecolor='blue', alpha=0.5)
    axes[1][0].set_title("GREEN VS NO GREEN WITH CULTURE")
    axes[1][0].set_xlabel(r"Initial attitude distance, $1-a_A/(a_A + b_A)$")
    axes[1][0].set_ylabel( r"Relative $\%$ change in final emissions")

    axes[1][1].plot(mean_list[::-1],mu_emissions_difference_matrix_compare_no_culture, ls="", marker=".", linewidth = 0.5, color='blue', label = r"Confirmation bias $\theta = %s$"% (theta))
    axes[1][1].fill_between(mean_list[::-1], min_emissions_difference_matrix_compare_no_culture, max_emissions_difference_matrix_compare_no_culture, facecolor='blue', alpha=0.5)
    axes[1][1].set_title("GREEN VS NO GREEN WITH CULTURE")
    axes[1][1].set_xlabel(r"Initial attitude distance, $1-a_A/(a_A + b_A)$")
    axes[1][1].set_ylabel( r"Relative $\%$ change in final emissions")

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_four_compare_%s" % (len(mean_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_emissions_no_culture_add_greens(fileName, emissions_list_no_culture_theta_one,emissions_list_culture_theta_one,  emissions_list_no_culture_theta_two,emissions_list_culture_theta_two, theta_one,theta_two,mean_list, dpi_save, seed_reps):
    fig, ax = plt.subplots(figsize=(10,7))    
    #IN THIS FUNCTION,  NO CULTURE = DEFAULT, CULTURE = ADD GREENS, CANT BE FUSSED TO CHANGE
    emissions_array_no_culture_theta_one = np.asarray(emissions_list_no_culture_theta_one)
    emissions_array_culture_theta_one = np.asarray(emissions_list_culture_theta_one)
    emissions_array_no_culture_theta_two = np.asarray(emissions_list_no_culture_theta_two)
    emissions_array_culture_theta_two = np.asarray(emissions_list_culture_theta_two)
    #print("emissions_array_culture",emissions_array_culture, emissions_array_culture.shape)

    emissions_difference_theta_one = ((emissions_array_culture_theta_one -  emissions_array_no_culture_theta_one )/emissions_array_no_culture_theta_one )*100
    emissions_difference_theta_two = ((emissions_array_culture_theta_two -  emissions_array_no_culture_theta_two )/emissions_array_no_culture_theta_two )*100

    mu_emissions_difference_theta_one = emissions_difference_theta_one.mean(axis=1)
    min_emissions_difference_theta_one = emissions_difference_theta_one.min(axis=1)
    max_emissions_difference_theta_one = emissions_difference_theta_one.max(axis=1)


    mu_emissions_difference_theta_two = emissions_difference_theta_two.mean(axis=1)
    min_emissions_difference_theta_two = emissions_difference_theta_two.min(axis=1)
    max_emissions_difference_theta_two = emissions_difference_theta_two.max(axis=1)

    #print("mu_emissions_difference ",mu_emissions_difference , mu_emissions_difference.shape)
    
    #quit()
    ax.plot(mean_list[::-1],mu_emissions_difference_theta_one, ls="", marker=".", linewidth = 0.5, color='blue', label = r"Confirmation bias $\theta = %s$"% (theta_one))
    ax.fill_between(mean_list[::-1], max_emissions_difference_theta_one, min_emissions_difference_theta_one, facecolor='blue', alpha=0.5)

    ax.plot(mean_list[::-1],mu_emissions_difference_theta_two, ls="", marker=".", linewidth = 0.5, color='red', label = r"Confirmation bias $\theta = %s$"% (theta_two))
    ax.fill_between(mean_list[::-1], max_emissions_difference_theta_two, min_emissions_difference_theta_two, facecolor='red', alpha=0.5)

    ax.set_xlabel(r"Initial attitude distance, $1-a_A/(a_A + b_A)$")
    ax.set_ylabel( r"Relative $\%$ change in final emissions")
    
    ax.set_title("NO GREEN VS GREEN WITHOUT CULTURE")
    
    ax.legend(loc = "lower right")
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_no_culture_add_greens_%s" % (len(mean_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_emissions_multi_ab_relative_all_add_green_two_theta(fileName, emissions_list_no_culture_theta_one,emissions_list_culture_theta_one,  emissions_list_no_culture_theta_two,emissions_list_culture_theta_two, theta_one,theta_two,mean_list, dpi_save, seed_reps):
    fig, ax = plt.subplots(figsize=(10,7))    

    
    emissions_array_no_culture_theta_one = np.asarray(emissions_list_no_culture_theta_one)
    emissions_array_culture_theta_one = np.asarray(emissions_list_culture_theta_one)
    emissions_array_no_culture_theta_two = np.asarray(emissions_list_no_culture_theta_two)
    emissions_array_culture_theta_two = np.asarray(emissions_list_culture_theta_two)
    #print("emissions_array_culture",emissions_array_culture, emissions_array_culture.shape)

    emissions_difference_theta_one = ((emissions_array_culture_theta_one -  emissions_array_no_culture_theta_one )/emissions_array_no_culture_theta_one )*100
    emissions_difference_theta_two = ((emissions_array_culture_theta_two -  emissions_array_no_culture_theta_two )/emissions_array_no_culture_theta_two )*100

    mu_emissions_difference_theta_one = emissions_difference_theta_one.mean(axis=1)
    min_emissions_difference_theta_one = emissions_difference_theta_one.min(axis=1)
    max_emissions_difference_theta_one = emissions_difference_theta_one.max(axis=1)


    mu_emissions_difference_theta_two = emissions_difference_theta_two.mean(axis=1)
    min_emissions_difference_theta_two = emissions_difference_theta_two.min(axis=1)
    max_emissions_difference_theta_two = emissions_difference_theta_two.max(axis=1)

    #print("mu_emissions_difference ",mu_emissions_difference , mu_emissions_difference.shape)
    
    #quit()
    ax.plot(mean_list[::-1],mu_emissions_difference_theta_one, ls="", marker=".", linewidth = 0.5, color='blue', label = r"Confirmation bias $\theta = %s$"% (theta_one))
    ax.fill_between(mean_list[::-1], max_emissions_difference_theta_one, min_emissions_difference_theta_one, facecolor='blue', alpha=0.5)

    ax.plot(mean_list[::-1],mu_emissions_difference_theta_two, ls="", marker=".", linewidth = 0.5, color='red', label = r"Confirmation bias $\theta = %s$"% (theta_two))
    ax.fill_between(mean_list[::-1], max_emissions_difference_theta_two, min_emissions_difference_theta_two, facecolor='red', alpha=0.5)

    ax.set_xlabel(r"Initial attitude distance, $1-a_A/(a_A + b_A)$")
    ax.set_ylabel( r"Relative $\%$ change in final emissions")

    ax.set_title("NO GREEN VS GREEN WITH CULTURE")
    ax.legend(loc = "lower right")
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_multi_ab_relative_all_add_green_two_theta_%s" % (len(mean_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")




def bifurcation_plot_add_green(fileName,emissions_pos_matrix,vals_list, dpi_save):
    fig, ax = plt.subplots()

    for i in range(len(vals_list)):
        x = [vals_list[i]]*(len(emissions_pos_matrix[i]))
        #print("vals_list[i]",vals_list[i])
        #print(x)
        y = emissions_pos_matrix[i]
        #print("y", y)
        
        #ax.scatter(x,y, color = "k")
        ax.plot(x,y, ls="", marker=".", color = "k", linewidth = 0.5)
        #ax.plot(x,y, ls="", color = "k")
    #ax.set_ylim(0,1)
    
    ax.set_xlabel(r"Initial attitude distance, $1-a_A/(a_A + b_A)$")
    ax.set_ylabel( r"$\%$ change in final emissions, $\Delta E_{\tau}$")
    
    plotName = fileName + "/Plots"
    f = plotName + "/bifurcation_plot_add_green_%s" % (len(vals_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_emissions_multi_ab_relative_all_add_green(fileName, emissions_list_default, emissions_list_add_green, mean_list, dpi_save, seed_reps):
    fig, ax = plt.subplots()

    emissions_array_add_green = np.asarray(emissions_list_add_green)
    emissions_array_default = np.asarray(emissions_list_default)
    #print("emissions_array_culture",emissions_array_culture, emissions_array_culture.shape)

    emissions_difference = ((emissions_array_add_green -  emissions_array_default)/emissions_array_default)*100

    #print("emissions_difference",emissions_difference, emissions_difference.shape)

    mu_emissions_difference = emissions_difference.mean(axis=1)
    sigma_emissions_difference = emissions_difference.std(axis=1)

    #print("mu_emissions_difference ",mu_emissions_difference , mu_emissions_difference.shape)
    
    #quit()
    # REVERSE IT FOR DISTANCE!
    ax.plot(mean_list[::-1],mu_emissions_difference[::-1], ls="", marker=".", linewidth = 0.5, color='blue')
    ax.fill_between(mean_list[::-1], mu_emissions_difference[::-1]+sigma_emissions_difference[::-1], mu_emissions_difference[::-1]-sigma_emissions_difference[::-1], facecolor='blue', alpha=0.5)

    ax.set_xlabel(r"Initial attitude distance, $1-a_A/(a_A + b_A)$")
    ax.set_ylabel( r"$\%$ change in final emissions, $\Delta E_{\tau}$")
    
    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_multi_ab_relative_all_add_green_%s" % (len(mean_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def live_print_culture_timeseries_green(
    fileName, Data_list, property_varied, title_list, nrows, ncols, dpi_save, colour_list
):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(10, 6), sharey=True)#, ,figsize=(14, 7)
    y_title = r"Identity, $I_{t,n}$"

    for i, ax in enumerate(axes.flat):
        for v in Data_list[i].agent_list:
            ax.plot(
                np.asarray(Data_list[i].history_time), np.asarray(v.history_culture), color = colour_list[i]
            )
            #print("v.history_culture",v.history_culture)

        ax.set_xlabel(r"Time")
        ax.set_ylim(0, 1.05)
        ax.set_title(title_list[i])

    axes[0].set_ylabel(r"%s" % y_title)

    plt.tight_layout()

    plotName = fileName + "/Prints"
    f = plotName + "/live_plot_culture_timeseries_green_%s.eps" % property_varied
    fig.savefig(f, dpi=dpi_save, format="eps")

def plot_compare_emissions_adding_green(fileName,Data_no_greens,Data_add_greens,dpi_save):
    
    fig, ax = plt.subplots()
    ax.plot(np.asarray(Data_no_greens.history_time), Data_no_greens.history_total_carbon_emissions, color =  "#4421af", linestyle='-', label = r"Default")
    ax.plot(np.asarray(Data_add_greens.history_time), Data_add_greens.history_total_carbon_emissions, color = "#5ad45a", linestyle="--", label = r"Green influencers")
    
    ax.legend()
    
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_compare_emissions_adding_green"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_behaviours_time_series_culture_and_emissions_ab_relative_all(fileName, data_list_culture, data_list_no_culture,emissions_list_culture, emissions_list_no_culture, mean_list, dpi_save, seed_reps):
    
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(14,7))

    axes[0].set_ylabel(r"Behavioural attitude, $A_{t,n,1}$")
    axes[0].set_title(r"Inter-behavioural dependence")
    axes[0].set_ylim(0, 1.05)

    axes[1].set_ylabel(r"Behavioural attitude, $A_{t,n,1}$")
    axes[1].set_title(r"Behavioural independence")
    axes[1].set_ylim(0, 1.05)

    axes[2].set_xlabel(r"Initial attitude mean, $a_A/(a_A + b_A)$")
    axes[2].set_ylabel( r" Relative $\%$ change in final emissions")
    
    axes[0].set_xlabel(r"Time")
    axes[1].set_xlabel(r"Time")
    #axes[2].set_xlabel(r"Time")

    ###########################################
    #plot cultue - black
    # label='Inter-behavioural dependance' if label == x_labels[0] else ''

    color_list = ["#B847A5", "#A5B847", "#47A5B8"]
    for i in range(len(data_list_culture)):
        for v in data_list_culture[i].agent_list:
            axes[0].plot(np.asarray(data_list_culture[i].history_time), np.asarray(v.history_behaviour_attitudes)[:,0], color = color_list[i], alpha= 0.5, linestyle="-")

    #########################################
    #plot no culture - red
    #label='Behavioural independance' if label == x_labels[0] else ''
    for i in range(len(data_list_culture)):
        for v in data_list_no_culture[i].agent_list:
            axes[1].plot(np.asarray(data_list_no_culture[i].history_time), np.asarray(v.history_behaviour_attitudes)[:,0], color = color_list[i], alpha= 0.5,linestyle="--")


    emissions_array_culture = np.asarray(emissions_list_culture)
    emissions_array_no_culture = np.asarray(emissions_list_no_culture)
    #print("emissions_array_culture",emissions_array_culture, emissions_array_culture.shape)

    emissions_difference = ((emissions_array_culture -  emissions_array_no_culture)/emissions_array_no_culture)*100

    #print("emissions_difference",emissions_difference, emissions_difference.shape)

    mu_emissions_difference = emissions_difference.mean(axis=1)
    sigma_emissions_difference = emissions_difference.std(axis=1)

    #print("mu_emissions_difference ",mu_emissions_difference , mu_emissions_difference.shape)
    
    #quit()
    axes[2].plot(mean_list,mu_emissions_difference, ls="", marker=".", linewidth = 0.5, color='blue')
    axes[2].fill_between(mean_list, mu_emissions_difference+sigma_emissions_difference, mu_emissions_difference-sigma_emissions_difference, facecolor='blue', alpha=0.5)
    
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_behaviours_time_series_culture_and_emissions_ab_relative_all"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_emissions_multi_ab_relative_all_two_theta(fileName, emissions_difference_theta_one, emissions_difference_theta_two, theta_one,theta_two,mean_list, dpi_save, seed_reps):
    fig, ax = plt.subplots()

    #print("emissions_difference",emissions_difference, emissions_difference.shape)

    mu_emissions_difference_theta_one = emissions_difference_theta_one.mean(axis=1)
    sigma_emissions_difference_theta_one = emissions_difference_theta_one.std(axis=1)

    mu_emissions_difference_theta_two = emissions_difference_theta_two.mean(axis=1)
    sigma_emissions_difference_theta_two = emissions_difference_theta_two.std(axis=1)

    #print("mu_emissions_difference ",mu_emissions_difference , mu_emissions_difference.shape)
    
    #quit()
    ax.plot(mean_list,mu_emissions_difference_theta_one, ls="", marker=".", linewidth = 0.5, color='blue', label = r"Confirmation bias $\theta = %s$"% (theta_one))
    ax.fill_between(mean_list, mu_emissions_difference_theta_one+sigma_emissions_difference_theta_one, mu_emissions_difference_theta_one-sigma_emissions_difference_theta_one, facecolor='blue', alpha=0.5)

    ax.plot(mean_list,mu_emissions_difference_theta_two, ls="", marker=".", linewidth = 0.5, color='red', label = r"Confirmation bias $\theta = %s$"% (theta_two))
    ax.fill_between(mean_list, mu_emissions_difference_theta_two+sigma_emissions_difference_theta_two, mu_emissions_difference_theta_two-sigma_emissions_difference_theta_two, facecolor='red', alpha=0.5)

    ax.set_xlabel(r"Initial attitude mean, $a_A/(a_A + b_A)$")
    ax.set_ylabel( r"$\%$ change in final emissions, $\Delta E_{\tau}$")
    
    ax.legend(loc = "upper left")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_multi_ab_relative_all_two_theta_%s" % (len(mean_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_emissions_multi_ab_relative_all_two_theta_reverse(fileName, emissions_difference_theta_one, emissions_difference_theta_two, theta_one,theta_two,mean_list, dpi_save, seed_reps):
    fig, ax = plt.subplots()

    #print("emissions_difference",emissions_difference, emissions_difference.shape)

    mu_emissions_difference_theta_one = emissions_difference_theta_one.mean(axis=1)

    sigma_emissions_difference_theta_one = emissions_difference_theta_one.std(axis=1)
    mu_emissions_difference_theta_two = emissions_difference_theta_two.mean(axis=1)
    sigma_emissions_difference_theta_two = emissions_difference_theta_two.std(axis=1)

    #print("mu_emissions_difference ",mu_emissions_difference , mu_emissions_difference.shape)
    
    #quit()
    ax.plot(mean_list[::-1],mu_emissions_difference_theta_one, ls="", marker=".", linewidth = 0.5, color='blue', label = r"Confirmation bias $\theta = %s$"% (theta_one))
    ax.fill_between(mean_list[::-1], mu_emissions_difference_theta_one+sigma_emissions_difference_theta_one, mu_emissions_difference_theta_one-sigma_emissions_difference_theta_one, facecolor='blue', alpha=0.5)

    ax.plot(mean_list[::-1],mu_emissions_difference_theta_two, ls="", marker=".", linewidth = 0.5, color='red', label = r"Confirmation bias $\theta = %s$"% (theta_two))
    ax.fill_between(mean_list[::-1], mu_emissions_difference_theta_two+sigma_emissions_difference_theta_two, mu_emissions_difference_theta_two-sigma_emissions_difference_theta_two, facecolor='red', alpha=0.5)

    ax.set_xlabel(r"Initial attitude distance, $1-a_A/(a_A + b_A)$")
    ax.set_ylabel( r"$\%$ change in final emissions, $\Delta E_{\tau}$")
    
    ax.legend(loc = "upper right")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_multi_ab_relative_all_two_theta_reverse_%s" % (len(mean_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_emissions_multi_ab_min_max_two_theta_reverse_add_green(fileName, emissions_difference_theta_one, emissions_difference_theta_two, theta_one,theta_two,mean_list, dpi_save, seed_reps):
    fig, ax = plt.subplots(figsize=(10,7))    

    mu_emissions_difference_theta_one = emissions_difference_theta_one.mean(axis=1)
    min_emissions_difference_theta_one = emissions_difference_theta_one.min(axis=1)
    max_emissions_difference_theta_one = emissions_difference_theta_one.max(axis=1)


    mu_emissions_difference_theta_two = emissions_difference_theta_two.mean(axis=1)
    min_emissions_difference_theta_two = emissions_difference_theta_two.min(axis=1)
    max_emissions_difference_theta_two = emissions_difference_theta_two.max(axis=1)

    #print("mu_emissions_difference ",mu_emissions_difference , mu_emissions_difference.shape)
    
    #quit()
    ax.plot(mean_list[::-1],mu_emissions_difference_theta_one, ls="", marker=".", linewidth = 0.5, color='blue', label = r"Confirmation bias $\theta = %s$"% (theta_one))
    ax.fill_between(mean_list[::-1], max_emissions_difference_theta_one, min_emissions_difference_theta_one, facecolor='blue', alpha=0.5)

    ax.plot(mean_list[::-1],mu_emissions_difference_theta_two, ls="", marker=".", linewidth = 0.5, color='red', label = r"Confirmation bias $\theta = %s$"% (theta_two))
    ax.fill_between(mean_list[::-1], max_emissions_difference_theta_two, min_emissions_difference_theta_two, facecolor='red', alpha=0.5)

    ax.set_xlabel(r"Initial attitude distance, $1-a_A/(a_A + b_A)$")
    ax.set_ylabel( r"Relative $\%$ change in final emissions")
    
    ax.legend(loc = "lower right")
    ax.set_title("CULTURE VS NO CULTURE WITH GREEN")
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_multi_ab_min_max_two_theta_reverse_add_green_%s" % (len(mean_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_emissions_multi_ab_min_max_two_theta_reverse(fileName, emissions_difference_theta_one, emissions_difference_theta_two, theta_one,theta_two,mean_list, dpi_save, seed_reps):
    fig, ax = plt.subplots()

    mu_emissions_difference_theta_one = emissions_difference_theta_one.mean(axis=1)
    min_emissions_difference_theta_one = emissions_difference_theta_one.min(axis=1)
    max_emissions_difference_theta_one = emissions_difference_theta_one.max(axis=1)


    mu_emissions_difference_theta_two = emissions_difference_theta_two.mean(axis=1)
    min_emissions_difference_theta_two = emissions_difference_theta_two.min(axis=1)
    max_emissions_difference_theta_two = emissions_difference_theta_two.max(axis=1)

    #print("mu_emissions_difference ",mu_emissions_difference , mu_emissions_difference.shape)
    
    #quit()
    ax.plot(mean_list[::-1],mu_emissions_difference_theta_one, ls="", marker=".", linewidth = 0.5, color='blue', label = r"Confirmation bias $\theta = %s$"% (theta_one))
    ax.fill_between(mean_list[::-1], max_emissions_difference_theta_one, min_emissions_difference_theta_one, facecolor='blue', alpha=0.5)

    ax.plot(mean_list[::-1],mu_emissions_difference_theta_two, ls="", marker=".", linewidth = 0.5, color='red', label = r"Confirmation bias $\theta = %s$"% (theta_two))
    ax.fill_between(mean_list[::-1], max_emissions_difference_theta_two, min_emissions_difference_theta_two, facecolor='red', alpha=0.5)

    ax.set_xlabel(r"Initial attitude distance, $1-a_A/(a_A + b_A)$")
    ax.set_ylabel( r"Relative $\%$ change in final emissions")
    
    ax.legend(loc = "upper right")
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_multi_ab_min_max_two_theta_reverse_%s" % (len(mean_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_emissions_multi_ab_relative_all(fileName, emissions_list_culture, emissions_list_no_culture, mean_list, dpi_save, seed_reps):
    fig, ax = plt.subplots()

    emissions_array_culture = np.asarray(emissions_list_culture)
    emissions_array_no_culture = np.asarray(emissions_list_no_culture)
    #print("emissions_array_culture",emissions_array_culture, emissions_array_culture.shape)

    emissions_difference = ((emissions_array_culture -  emissions_array_no_culture)/emissions_array_no_culture)*100

    #print("emissions_difference",emissions_difference, emissions_difference.shape)

    mu_emissions_difference = emissions_difference.mean(axis=1)
    sigma_emissions_difference = emissions_difference.std(axis=1)

    #print("mu_emissions_difference ",mu_emissions_difference , mu_emissions_difference.shape)
    
    #quit()
    ax.plot(mean_list,mu_emissions_difference, ls="", marker=".", linewidth = 0.5, color='blue')
    ax.fill_between(mean_list, mu_emissions_difference+sigma_emissions_difference, mu_emissions_difference-sigma_emissions_difference, facecolor='blue', alpha=0.5)

    ax.set_xlabel(r"Initial attitude mean, $a_A/(a_A + b_A)$")
    ax.set_ylabel( r"$\%$ change in final emissions, $\Delta E_{\tau}$")
    
    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_multi_ab__relative_%s" % (len(mean_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_emissions_multi_ab_all(fileName, emissions_list_culture, emissions_list_no_culture, mean_list, dpi_save, seed_reps):
    fig, ax = plt.subplots()

    ################################################
    #culture

    ax.plot(mean_list,emissions_list_culture, ls="", marker=".", color = "k", linewidth = 0.5)
    #######################################
    #no culture
    ax.plot(mean_list,emissions_list_no_culture, ls="", marker=".", color = "red", linewidth = 0.5)
    
    ax.set_xlabel(r"Initial attitude mean, $a_A/(a_A + b_A)$")
    ax.set_ylabel(r"Total emissions, $E_t$")
    
    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_multi_ab_%s" % (len(mean_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_emissions_multi_ab_relative(fileName, emissions_list_culture, emissions_list_no_culture, mean_list, dpi_save):
    fig, ax = plt.subplots()

    emissions_difference = np.asarray(emissions_list_culture) -  np.asarray(emissions_list_no_culture)

    ax.plot(mean_list,emissions_difference, ls="", marker=".", color = "k", linewidth = 0.5)
    
    ax.set_xlabel(r"Initial attitude mean, $a_A/(a_A + b_A)$")
    ax.set_ylabel( r"Change in final emissions, $\Delta E_{\tau}$")
    
    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_multi_ab__relative_%s" % (len(mean_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_emissions_multi_ab(fileName, emissions_list_culture, emissions_list_no_culture, mean_list, dpi_save):
    fig, ax = plt.subplots()

    ################################################
    #culture

    ax.plot(mean_list,emissions_list_culture, ls="", marker=".", color = "k", linewidth = 0.5)
    #######################################
    #no culture
    ax.plot(mean_list,emissions_list_no_culture, ls="", marker=".", color = "red", linewidth = 0.5)
    
    ax.set_xlabel(r"Initial attitude mean, $a_A/(a_A + b_A)$")
    ax.set_ylabel(r"Total emissions, $E_t$")
    
    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_multi_ab_%s" % (len(mean_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_emissions_distance(fileName,data_list_culture, data_list_no_culture,init_attitudes_list, dpi_save):

    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(14,7))
    
    #axes[2].get_shared_y_axes().remove(axes[2])

    axes[0].set_ylabel(r"Behavioural attitude, $A_{t,n,1}$")
    axes[0].set_title(r"Inter-behavioural dependence")
    axes[0].set_ylim(0, 1)

    axes[1].set_ylabel(r"Behavioural attitude, $A_{t,n,1}$")
    axes[1].set_title(r"Behavioural independence")
    axes[1].set_ylim(0, 1)

    axes[2].set_ylabel(r"Total emissions, $E_t$")
    
    axes[0].set_xlabel(r"Time")
    axes[1].set_xlabel(r"Time")
    axes[2].set_xlabel(r"Time")

    ###########################################
    #plot cultue - black
    # label='Inter-behavioural dependance' if label == x_labels[0] else ''

    color_list = ["red", "green", "blue"]
    for i in range(len(data_list_culture)):
        for v in data_list_culture[i].agent_list:
            axes[0].plot(np.asarray(data_list_culture[i].history_time), np.asarray(v.history_behaviour_attitudes)[:,0], color = color_list[i], alpha= 0.5, linestyle="-")

    #########################################
    #plot no culture - red
    #label='Behavioural independance' if label == x_labels[0] else ''
    for i in range(len(data_list_culture)):
        for v in data_list_no_culture[i].agent_list:
            axes[1].plot(np.asarray(data_list_no_culture[i].history_time), np.asarray(v.history_behaviour_attitudes)[:,0], color = color_list[i], alpha= 0.5,linestyle="--")

    #plot culture - black
    for i in range(len(data_list_culture)):
        axes[2].plot(np.asarray(data_list_culture[i].history_time), data_list_culture[i].history_total_carbon_emissions, color = color_list[i], linestyle='-', label = r"Inter-behavioural dependence, $a_A = %s, b_A = %s$" % (init_attitudes_list[i][0], init_attitudes_list[i][1]))
        axes[2].plot(np.asarray(data_list_no_culture[i].history_time), data_list_no_culture[i].history_total_carbon_emissions, color = color_list[i], linestyle="--", label = r"Behavioural independence, $a_A = %s, b_A = %s$" % (init_attitudes_list[i][0], init_attitudes_list[i][1]))
    
    axes[2].legend(loc = 'upper left', prop={'size': 11.5})
    
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_emissions_distance"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_behaviours_time_series_emissions_and_behaviour(fileName,Data_culture, Data_no_culture, dpi_save):

    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(14,7))
    
    #axes[2].get_shared_y_axes().remove(axes[2])

    axes[0].set_ylabel(r"Behavioural attitude, $A_{t,n,1}$")
    axes[0].set_title(r"Inter-behavioural dependence")
    axes[0].set_ylim(0, 1)

    axes[1].set_ylabel(r"Behavioural attitude, $A_{t,n,1}$")
    axes[1].set_title(r"Behavioural independence")
    axes[1].set_ylim(0, 1)

    axes[2].set_ylabel(r"Total emissions, $E_t$")
    
    axes[0].set_xlabel(r"Time")
    axes[1].set_xlabel(r"Time")
    axes[2].set_xlabel(r"Time")

    ###########################################
    #plot cultue - black
    # label='Inter-behavioural dependance' if label == x_labels[0] else ''
    for v in Data_culture.agent_list:
        axes[0].plot(np.asarray(Data_culture.history_time), np.asarray(v.history_behaviour_attitudes)[:,0], color = "k", alpha= 0.5, linestyle="-")

    #########################################
    #plot no culture - red
    #label='Behavioural independance' if label == x_labels[0] else ''
    for v in Data_no_culture.agent_list:
        axes[1].plot(np.asarray(Data_no_culture.history_time), np.asarray(v.history_behaviour_attitudes)[:,0], color = "red", alpha= 0.5,linestyle="--")

    #plot culture - black
    axes[2].plot(np.asarray(Data_culture.history_time), Data_culture.history_total_carbon_emissions,color="k",linestyle='-', label = "Inter-behavioural dependence")

    axes[2].plot(np.asarray(Data_no_culture.history_time), Data_no_culture.history_total_carbon_emissions,color="red",linestyle="--", label = "Behavioural independence")
    axes[2].legend(loc = 'upper left', prop={'size': 11.5})
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_behaviours_time_series_emissions_and_beahviour"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")



def plot_behaviours_time_series_emissions_and_culture(fileName,Data_culture, Data_no_culture, dpi_save):
    fig, axes = plt.subplots(nrows = 1, ncols = Data_culture.M + 1, figsize=(14,7), sharey=True)
    
    axes[0].set_ylabel(r"Behavioural attitude, $A_{t,n,m}$")
    axes[0].set_ylim(0, 1)

    axes[Data_culture.M].get_shared_y_axes().remove(axes[Data_culture.M])
    
    print(axes)
    print(axes[0].get_shared_y_axes())
    print(axes[Data_culture.M].get_shared_y_axes())
    #axes.get_shared_y_axes().remove(axes[Data_culture.M + 1])

    for z, ax in enumerate(axes.flat):
        if z < Data_culture.M:
            #########################################
            #plot no culture - red
            #label='Behavioural independance' if label == x_labels[0] else ''
            for v in Data_no_culture.agent_list:
                ax.plot(np.asarray(Data_no_culture.history_time), np.asarray(v.history_behaviour_attitudes)[:,z], color = "red", alpha= 0.5 )

            ###########################################
            #plot cultue - black
            # label='Inter-behavioural dependance' if label == x_labels[0] else ''
            for v in Data_culture.agent_list:
                ax.plot(np.asarray(Data_culture.history_time), np.asarray(v.history_behaviour_attitudes)[:,z], color = "k", alpha= 0.5)


            ax.set_title(r"$\phi_{%s} = %s$" % (z + 1, Data_culture.phi_array[z]))
            ax.set_xlabel(r"Time")
        else:
            
            #########################################
            #plot no culture - red
            #label='Behavioural independance' if label == x_labels[0] else ''
            for v in Data_no_culture.agent_list:
                ax.plot(np.asarray(Data_no_culture.history_time), np.asarray(v.history_culture), color = "red", alpha= 0.5 )

            ###########################################
            #plot cultue - black
            # label='Inter-behavioural dependance' if label == x_labels[0] else ''
            for v in Data_culture.agent_list:
                ax.plot(np.asarray(Data_culture.history_time), np.asarray(v.history_culture), color = "k", alpha= 0.5)

            ax.set_xlabel(r"Time")
            ax.set_ylabel(r"Identity, $I_{t,n}$")
            ax.set_ylim(0, 1)

            ########################
            # Now do the emissions over time
            ax2=ax.twinx()
            # twin object for two different y-axis on the sample plot
            #plot no culture - red
            ax2.plot(np.asarray(Data_no_culture.history_time), Data_no_culture.history_total_carbon_emissions,color="red",linestyle="--")
            #plot culture - black
            ax2.plot(np.asarray(Data_culture.history_time), Data_culture.history_total_carbon_emissions,color="k",linestyle='--')

            ax2.set_ylabel(r"Total emissions, $E$")

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_behaviours_time_series_emissions_and_culture"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_behaviours_time_series_emissions(fileName,Data_culture, Data_no_culture, dpi_save):
    fig, axes = plt.subplots(nrows = 1, ncols = Data_culture.M, figsize=(10,6), sharey=True)
    
    axes[0].set_ylabel(r"Behavioural attitude, $A_{t,n,m}$")
    axes[0].set_ylim(0, 1)

    for z, ax in enumerate(axes.flat):
        #########################################
        #plot no culture - red
        #label='Behavioural independance' if label == x_labels[0] else ''
        for v in Data_no_culture.agent_list:
            ax.plot(np.asarray(Data_no_culture.history_time), np.asarray(v.history_behaviour_attitudes)[:,z], color = "red", alpha= 0.5 )

        ###########################################
        #plot cultue - black
        # label='Inter-behavioural dependance' if label == x_labels[0] else ''
        for v in Data_culture.agent_list:
            ax.plot(np.asarray(Data_culture.history_time), np.asarray(v.history_behaviour_attitudes)[:,z], color = "k", alpha= 0.5)


        ax.set_title(r"$\phi_{%s} = %s$" % (z + 1, Data_culture.phi_array[z]))
        ax.set_xlabel(r"Time")

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_behaviours_time_series_emissions"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_culture_time_series_emissions(fileName,Data_culture, Data_no_culture, dpi_save):
    fig, ax = plt.subplots(figsize=(10,6))

    ###########################################
    #plot cultue - black
    # label='Inter-behavioural dependance' if label == x_labels[0] else ''
    for v in Data_culture.agent_list:
        ax.plot(np.asarray(Data_culture.history_time), np.asarray(v.history_culture), color = "k", alpha= 0.5)

    #########################################
    #plot no culture - red
    #label='Behavioural independance' if label == x_labels[0] else ''
    for v in Data_no_culture.agent_list:
        ax.plot(np.asarray(Data_no_culture.history_time), np.asarray(v.history_culture), color = "red", alpha= 0.5 )

    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Identity, $I_{t,n}$")
    ax.set_ylim(0, 1.01)
    

    ########################
    # Now do the emissions over time
    ax2=ax.twinx()
    # twin object for two different y-axis on the sample plot
    #plot culture - black
    ax2.plot(np.asarray(Data_culture.history_time), Data_culture.history_total_carbon_emissions,color="k",linestyle='--')
    #plot no culture - red
    ax2.plot(np.asarray(Data_no_culture.history_time), Data_no_culture.history_total_carbon_emissions,color="red",linestyle="--")

    ax2.set_ylabel(r"Total emissions, $E$")


    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_culture_time_series_emissions"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

####bifurcation plots
def bifurcation_plot(fileName,cluster_pos_matrix,vals_list, dpi_save):
    fig, ax = plt.subplots()

    for i in range(len(vals_list)):
        x = [vals_list[i]]*(len(cluster_pos_matrix[i]))
        #print("vals_list[i]",vals_list[i])
        #print(x)
        y = cluster_pos_matrix[i]
        #print("y", y)
        
        #ax.scatter(x,y, color = "k")
        ax.plot(x,y, ls="", marker=".", color = "k", linewidth = 0.5)
        #ax.plot(x,y, ls="", color = "k")
    ax.set_ylim(0,1)
    
    ax.set_xlabel(r"Confirmation Bias")
    ax.set_ylabel(r"Final identity cluster")
    
    plotName = fileName + "/Plots"
    f = plotName + "/bifurcation_plot_%s" % (len(vals_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def bifurcation_plot_culture_or_not(fileName,cluster_pos_matrix_identity,cluster_pos_matrix_no_identity,vals_list_identity,vals_list_no_identity, dpi_save):
    fig, axes = plt.subplots(nrows = 1, ncols=2, sharey= True, figsize= (10,6))

    print(axes) 

    for i in range(len(vals_list_identity)):
        x_identity = [vals_list_identity[i]]*(len(cluster_pos_matrix_identity[i]))
        #print("vals_list[i]",vals_list[i])
        #print(x)
        y_identity = cluster_pos_matrix_identity[i]
        #print("y", y)
        
        #ax.scatter(x,y, color = "k")
        axes[0].plot(x_identity,y_identity, ls="", marker=".", color = "k", linewidth = 0.5)
        #ax.plot(x,y, ls="", color = "k")
    
    for i in range(len(vals_list_no_identity)):
        x_no_identity = [vals_list_no_identity[i]]*(len(cluster_pos_matrix_no_identity[i]))
        #print("vals_list[i]",vals_list[i])
        #print(x)
        y_no_identity = cluster_pos_matrix_no_identity[i]
        #print("y", y)
        
        #ax.scatter(x,y, color = "k")
        axes[1].plot(x_no_identity,y_no_identity, ls="", marker=".", color = "r", linewidth = 0.5)
        #ax.plot(x,y, ls="", color = "k")

    axes[0].set_ylim(0,1)

    axes[0].set_title(r"Inter-behavioural dependance")
    axes[1].set_title(r"Behavioural independance")

    axes[0].set_xlabel(r"Confirmation bias, $\theta$")
    axes[1].set_xlabel(r"Confirmation bias, $\theta$")
    axes[0].set_ylabel(r"Final identity clusters")
    
    plotName = fileName + "/Plots"
    f = plotName + "/bifurcation_plot_%s_%s" % (len(vals_list_identity),len(vals_list_no_identity))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def bifurcation_plot_stochastic(fileName,cluster_pos_matrix_list,vals_list, seed_list,cmap,dpi_save):
    fig, ax = plt.subplots()
    #print(seed_list)
    #colour_adjust = Normalize(min(seed_list), max(seed_list))
    #step_colours = cmap(colour_adjust)

    for i in range(len(cluster_pos_matrix_list)):
        for j in range(len(vals_list)):
            x = [vals_list[j]]*(len(cluster_pos_matrix_list[i][j]))
            #print("vals_list[i]",vals_list[i])
            #print(x)
            y = cluster_pos_matrix_list[i][j]
            #print("y", y)
            
            #ax.scatter(x,y, color = "k")
            ax.plot(x,y, ls="", marker=".", linewidth = 0.5)# color = step_colours[j],
            #ax.plot(x,y, ls="", color = "k")
    ax.set_ylim(0,1)
    
    ax.set_xlabel(r"Confirmation Bias")
    ax.set_ylabel(r"Final identity cluster")
    
    plotName = fileName + "/Plots"
    f = plotName + "/bifurcation_plot_stochastic_%s" % (len(vals_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def bifurcation_heat_map_stochastic(fileName,cluster_pos_matrix_list,vals_list, seed_list,cmap,dpi_save):
    
    fig, ax = plt.subplots()
    #print(seed_list)
    #colour_adjust = Normalize(min(seed_list), max(seed_list))
    #step_colours = cmap(colour_adjust)

    x_list = []
    y_list = []
    for i in range(len(cluster_pos_matrix_list)):
        for j in range(len(vals_list)):
            y_list.extend(cluster_pos_matrix_list[i][j])
            x_list.extend([vals_list[j]]*(len(cluster_pos_matrix_list[i][j])))

            
            #ax.scatter(x,y, color = "k")
            #ax.plot(x,y, ls="", marker=".", linewidth = 0.5)# color = step_colours[j],
            #ax.plot(x,y, ls="", color = "k")
    print("x_list", x_list)
    print("y_list", y_list)
    #flatten the list
    #print(len(x_list), len(y_list))
    H, xedges, yedges = np.histogram2d(x_list, y_list, bins=(100,100))

    #pcolormesh
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H)

    #interpolated image
    #im = NonUniformImage(ax, interpolation='bilinear')
    #xcenters = (xedges[:-1] + xedges[1:]) / 2
    #ycenters = (yedges[:-1] + yedges[1:]) / 2
    #im.set_data(xcenters, ycenters, H)
    #ax.images.append(im)

    #ax.set_ylim(0,1)
    
    ax.set_xlabel(r"Confirmation Bias")
    ax.set_ylabel(r"Final identity cluster")
    
    plotName = fileName + "/Plots"
    f = plotName + "/bifurcation_heatmap_stochastic_%s" % (len(vals_list))
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def bifurcation_plot_one_seed_two_params(fileName,cluster_pos_matrix_list,variable_parameters_dict,cmap,dpi_save):
    fig, ax = plt.subplots()
    
    colour_adjust = Normalize(variable_parameters_dict["row"]["vals"])
    step_colours = cmap(colour_adjust)

    for i in range(variable_parameters_dict["row"]["reps"]):
        for j in range(variable_parameters_dict["col"]["reps"]):

            x = [variable_parameters_dict["row"]["vals"][j]]*(len(cluster_pos_matrix_list[i][j]))#confirmation bias
            y = cluster_pos_matrix_list[i][j]#identity
            z = [variable_parameters_dict["col"]["vals"][j]]*(len(cluster_pos_matrix_list[i][j]))#polarisation

            ax.plot(x,y,x, ls="", marker=".", color = step_colours[j], linewidth = 0.5)#colour based on polarsation
            #ax.plot(x,y, ls="", color = "k")
    ax.set_ylim(0,1)
    
    ax.set_xlabel(r"Confirmation Bias")
    ax.set_ylabel(r"Final identity cluster")
    ax.set_zlabel(r"Initial polarisation")
    
    plotName = fileName + "/Plots"
    f = plotName + "/bifurcation_plot_one_seed_two_params_%s_%s" % (variable_parameters_dict["row"]["reps"],variable_parameters_dict["col"]["reps"])
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

###### estimating number of clusters
def calc_num_clusters_auto_bandwidth_return(culture_data, s):
    
    kde = gaussian_kde(culture_data)
    probs = kde.evaluate(s)#UsES scott's rule
    #ma_scipy = argrelextrema(probs, np.greater)[0]
    return  kde, probs #ma_scipylen(ma_scipy),

def calc_num_clusters_set_bandwidth(culture_data,s,bandwidth):
    X_reshape = culture_data.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X_reshape)
    e = kde.score_samples(s.reshape(-1,1))
    return kde, e

def cluster_estimation_plot(Data,s,bandwidth):
    culture_data = np.asarray([Data.agent_list[n].culture for n in range(Data.N)])

    cluster_count, kde, probs, ma_scipy = calc_num_clusters_auto_bandwidth_return(culture_data, s)
    ma_scipy = argrelextrema(probs, np.greater)[0]
    print("ma auto bandwidth", ma_scipy)

    kde, e = calc_num_clusters_set_bandwidth(culture_data, s, bandwidth)
    ma = argrelextrema(e, np.greater)[0]
    print("ma set bandwidth", ma)

    fig, ax = plt.subplots()
    ax.plot(s, probs, label = "Auto bandwidth")
    ax.plot(s, e, label = "Set bandwidth")
    ax.legend()

def MeanShift_clusters(X,bandwidth):
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)

    #print("number of estimated clusters : %d" % n_clusters)
    return ms, labels, cluster_centers, labels_unique, n_clusters

def plot_alpha_group(fileName, Data, dpi_save,s, auto_bandwidth, bandwidth,cmap,  round_dec, norm_zero_one):

    culture_data = np.asarray([Data.agent_list[n].culture for n in range(Data.N)])

    if auto_bandwidth:
        kde, e = calc_num_clusters_auto_bandwidth_return(culture_data, s)
        bandwidth = kde.factor
    else:
        kde, e = calc_num_clusters_set_bandwidth(culture_data,s,bandwidth)
    
    print("bandwidth used:",bandwidth)

    mi = argrelextrema(e, np.less)[0]#list of minimum values in the kde
    ma = argrelextrema(e, np.greater)[0]#list of minimum values in the kde

    clusters_index_lists = get_cluster_list(culture_data,s, Data.N, mi)
    #print(a[a >= s[mi][-1]])  

    print("clusters_index_lists",clusters_index_lists)

    ########################################################
    #NOW plot the convergence
    fig, ax = plt.subplots()
    
    #get a list of values that i think it shoudl vonverge too 1/N_G

    inverse_N_g_list = [1/len(i) for i in clusters_index_lists]
    #print("inverse_N_g_list",inverse_N_g_list)

    #quit()
    time_vals_data = []
    for t in range(len(Data.history_time)):
        time_vals_data_row = []
        for i in range(len(clusters_index_lists)):
            #print("clusters_index_lists[i]",clusters_index_lists[i],len(clusters_index_lists[i]))
            sub_weighting_matrix = Data.history_weighting_matrix[t][clusters_index_lists[i]]
            #print("sub_weighting_matrix", sub_weighting_matrix, sub_weighting_matrix.shape)
            #i want a matrix that excludes all the values that arent from the indes in the clusters_index_lists[i]
            sub_sub_weighting_matrix = sub_weighting_matrix[:,clusters_index_lists[i]]
            #print("sub_sub_weighting_matrix", sub_sub_weighting_matrix, sub_sub_weighting_matrix.shape)

            mean_weighting_val = np.mean(sub_sub_weighting_matrix)
            #print("mean_value",mean_weighting_val)

            time_vals_data_row.append(mean_weighting_val)
            #quit()
        time_vals_data.append(time_vals_data_row)
    
    time_vals_data_array = np.asarray(time_vals_data)
    #print("time_vals_data_array", time_vals_data_array.shape)
    vals_time_data = time_vals_data_array.T
    #print("vals_time_data ",vals_time_data.shape)
    
    cluster_example_identity_list = s[ma]
    colour_adjust = norm_zero_one(cluster_example_identity_list)
    ani_step_colours = cmap(colour_adjust)
    #print(ani_step_colours)

    for i in range(len(clusters_index_lists)): 
        ax.plot(Data.history_time, vals_time_data[i], color = ani_step_colours[i])#label = "Cluster %s" % (i)
        ax.axhline(y= inverse_N_g_list[i], color = ani_step_colours[i], linestyle = "--")

    ax.set_title("Bandwidth = %s" % round(bandwidth, round_dec) )
    #ax.legend()

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm_zero_one), ax=ax
    )
    cbar.set_label(r"Identity, $I_{t,n}$")


    plotName = fileName + "/Plots"
    f = plotName + "/plot_alpha_group_auto_bandwidth_bool_%s_%s" % (auto_bandwidth, bandwidth)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

    return clusters_index_lists,cluster_example_identity_list

def plot_alpha_group_multi(fileName, Data_list, dpi_save,s, auto_bandwidth, bandwidth,cmap, norm_zero_one, nrows, ncols,title_list):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    clusters_index_lists_list = []
    cluster_example_identity_list_list = []

    for z, ax in enumerate(axes.flat):
        culture_data = np.asarray([Data_list[z].agent_list[n].culture for n in range(Data_list[z].N)])

        if auto_bandwidth:
            kde, e = calc_num_clusters_auto_bandwidth_return(culture_data, s)
            bandwidth = kde.factor
        else:
            kde, e = calc_num_clusters_set_bandwidth(culture_data,s,bandwidth)
        
        print("bandwidth used:",bandwidth)

        mi = argrelextrema(e, np.less)[0]#list of minimum values in the kde
        ma = argrelextrema(e, np.greater)[0]#list of minimum values in the kde
        
        if len(s[mi]) == 0:
            print("NO PEAKS?", s[mi])
            clusters_index_lists_list.append([])
            cluster_example_identity_list_list.append([])
        else:
            clusters_index_lists = get_cluster_list(culture_data,s, Data_list[z].N, mi)
            clusters_index_lists_list.append(clusters_index_lists)
            #print(a[a >= s[mi][-1]])  

            #print("clusters_index_lists",clusters_index_lists)
            
            #get a list of values that i think it shoudl vonverge too 1/N_G

            inverse_N_g_list = [1/len(i) for i in clusters_index_lists]
            #print("inverse_N_g_list",inverse_N_g_list)

            #quit()
            time_vals_data = []
            for t in range(len(Data_list[z].history_time)):
                time_vals_data_row = []
                for i in range(len(clusters_index_lists)):
                    #print("clusters_index_lists[i]",clusters_index_lists[i],len(clusters_index_lists[i]))
                    sub_weighting_matrix = Data_list[z].history_weighting_matrix[t][clusters_index_lists[i]]
                    #print("sub_weighting_matrix", sub_weighting_matrix, sub_weighting_matrix.shape)
                    #i want a matrix that excludes all the values that arent from the indes in the clusters_index_lists[i]
                    sub_sub_weighting_matrix = sub_weighting_matrix[:,clusters_index_lists[i]]
                    #print("sub_sub_weighting_matrix", sub_sub_weighting_matrix, sub_sub_weighting_matrix.shape)

                    mean_weighting_val = np.mean(sub_sub_weighting_matrix)
                    #print("mean_value",mean_weighting_val)

                    time_vals_data_row.append(mean_weighting_val)
                    #quit()
                time_vals_data.append(time_vals_data_row)
            
            time_vals_data_array = np.asarray(time_vals_data)
            #print("time_vals_data_array", time_vals_data_array.shape)
            vals_time_data = time_vals_data_array.T
            #print("vals_time_data ",vals_time_data.shape)
        
            ########################################################
            #NOW plot the convergence
        

            cluster_example_identity_list = s[ma]
            cluster_example_identity_list_list.append(cluster_example_identity_list)
            colour_adjust = norm_zero_one(cluster_example_identity_list)
            ani_step_colours = cmap(colour_adjust)
            #print(ani_step_colours)

            for i in range(len(clusters_index_lists)): 
                ax.plot(Data_list[z].history_time, vals_time_data[i], color = ani_step_colours[i])#label = "Cluster %s" % (i)
                ax.axhline(y= inverse_N_g_list[i], color = ani_step_colours[i], linestyle = "--")

            ax.set_title(title_list[z])
            #ax.legend()

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm_zero_one), ax=axes.ravel().tolist()
    )
    cbar.set_label(r"Identity, $I_{t,n}$")

    plotName = fileName + "/Prints"
    f = plotName + "/ plot_alpha_group_multi_%s_%s" % (auto_bandwidth, bandwidth)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

    return clusters_index_lists_list,cluster_example_identity_list_list

def plot_cluster_culture_time_series(fileName, Data, dpi_save,clusters_index_lists,cluster_example_identity_list, cmap,norm_zero_one, shuffle_colours):
    fig, ax = plt.subplots(figsize=(10,6))
    y_title = r"Identity, $I_{t,n}$"
    
    #colour_adjust = norm_zero_one(cluster_example_identity_list)
    #ani_step_colours = cmap(colour_adjust)

    cmap = get_cmap(name='hsv', lut = len(cluster_example_identity_list))
    ani_step_colours = [cmap(i) for i in range(len(cluster_example_identity_list))] 
    if shuffle_colours:
        np.random.shuffle(ani_step_colours)
    else:
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=norm_zero_one), ax=ax
        )
        cbar.set_label(r"Cluster center Identity, $I_{t,n}$")
    #print("ani_step_colours",ani_step_colours)

    colours_dict = {}#It cant be a list as you need to do it out of order
    for i in range(len(clusters_index_lists)):#i is the list of index in that cluster
        for j in clusters_index_lists[i]:#j is an index in that cluster
            #print(i,j)
            colours_dict["%s" % (j)] = ani_step_colours[i]
        
    print("colours_dict",colours_dict)

    for v in range(len(Data.agent_list)):
        ax.plot(np.asarray(Data.history_time), np.asarray(Data.agent_list[v].history_culture), color = colours_dict["%s" % (v)])
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_ylim(0, 1)

    #plt.tight_layout()


    plotName = fileName + "/Plots"
    f = plotName + "/plot_cluster_culture_time_series_suffel_bool-%s" % (shuffle_colours)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_cluster_culture_time_series_multi(fileName, Data_list, dpi_save,clusters_index_lists_list,cluster_example_identity_list_list, cmap,norm_zero_one, shuffle_colours, nrows, ncols,title_list):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, constrained_layout=True)

    print("clusters_index_lists_list",len(clusters_index_lists_list))
    print("cluster_example_identity_list_list",len(cluster_example_identity_list_list))

    for z, ax in enumerate(axes.flat):
        y_title = r"Identity, $I_{t,n}$"
        
        #colour_adjust = norm_zero_one(cluster_example_identity_list)
        #ani_step_colours = cmap(colour_adjust)
        if len(cluster_example_identity_list_list[z]) == 0:
            print("No cluster")
            for v in range(len(Data_list[z].agent_list)):
                ax.plot(np.asarray(Data_list[z].history_time), np.asarray(Data_list[z].agent_list[v].history_culture))
                ax.set_xlabel(r"Time")
                ax.set_ylabel(r"%s" % y_title)
                ax.set_ylim(0, 1)

                ax.set_title("No Cluster, " + title_list[z])
        else:
            cmap = get_cmap(name='hsv', lut = len(cluster_example_identity_list_list[z]))
            ani_step_colours = [cmap(i) for i in range(len(cluster_example_identity_list_list[z]))] 

            if shuffle_colours:
                np.random.shuffle(ani_step_colours)
            else:
                cbar = fig.colorbar(
                    plt.cm.ScalarMappable(cmap=cmap, norm=norm_zero_one), ax=ax
                )
                cbar.set_label(r"Cluster center Identity, $I_{t,n}$")
            #print("ani_step_colours",ani_step_colours)

            colours_dict = {}#It cant be a list as you need to do it out of order
            for i in range(len(clusters_index_lists_list[z])):#i is the list of index in that cluster
                for j in clusters_index_lists_list[z][i]:#j is an index in that cluster
                    #print(i,j)
                    colours_dict["%s" % (j)] = ani_step_colours[i]
                
            #print("colours_dict",colours_dict)

            for v in range(len(Data_list[z].agent_list)):
                ax.plot(np.asarray(Data_list[z].history_time), np.asarray(Data_list[z].agent_list[v].history_culture), color = colours_dict["%s" % (v)])
                ax.set_xlabel(r"Time")
                ax.set_ylabel(r"%s" % y_title)
                ax.set_ylim(0, 1)

            ax.set_title(title_list[z])

            #plt.tight_layout()

    plotName = fileName + "/Prints"
    f = plotName + "/plot_cluster_culture_time_series_multi_suffel_bool-%s" % (shuffle_colours)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_last_culture_vector_matrix(
    fileName, Data_list, dpi_save, property_varied, property_varied_title, property_varied_vals
):
    fig, ax = plt.subplots()

    Z = []
    for i in range(len(Data_list)):
        Z.append(Data_list[i].culture_list)

    Z_i_val = np.asarray(Z).T
    #print(Z_i_val[0], Z_i_val.shape)
    #ax.matshow(
    #    Z,
    #    cmap=cmap,
    #    aspect="auto",
    #)
    for i in range(len(Z)):
        ax.scatter(x = property_varied_vals, y = Z_i_val[i])
    #better as a scatter plot? or density
    # what about a contour map? 
    ax.set_xlabel(property_varied_title)
    ax.set_ylabel(r"Identity, $I_{t,n}$")

    # colour bar axes
    #cbar = fig.colorbar(
    #    plt.cm.ScalarMappable(cmap=cmap, norm=norm_zero_one),
    #    ax=ax
    #)
    #cbar.set_label(r"Social network weighting, $\alpha_{n,k}$", labelpad= 5)

    plotName = fileName + "/Plots"
    f = (
        plotName
        + "/plot_last_culture_vector_matrix_%s.png"
        % property_varied
    )
    fig.savefig(f, dpi=dpi_save, format="png")
    # fig.savefig(f, dpi=dpi_save,format='eps')  


#####RUNPLOT PLOTS - SINGLE NETWORK
def live_print_culture_timeseries_varynetwork_structure(
    fileName,
    Data_dict,
    nrows,
    ncols,
    dpi_save,
):

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(14, 7), constrained_layout=True
    )

    y_title = r"Identity, $I_{t,n}$"

    structure_list = list(Data_dict.keys())
    structure_data = list(Data_dict.values())
    
    structure_list_title = ["Watts-Strogatz small world", "Barabasi-Albert scale free"]
    
    for i, ax in enumerate(axes.flat):
        for v in structure_data[i].agent_list:
            ax.plot(
                np.asarray(structure_data[i].history_time), np.asarray(v.history_culture)
            )

        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title(structure_list_title[i])
        #ax.set_ylim(0, 1)

    plotName = fileName + "/Prints"
    f = plotName + "/live_print_culture_timeseries_varynetwork_structure_%s" % (
        len(structure_list),
    )
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def draw_networks(
    fileName,
    Data_dict,
    nrows,
    ncols,
    dpi_save,
    norm_zero_one,
    cmap,
):

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(14, 7), constrained_layout=True
    )

    structure_list = list(Data_dict.keys())
    structure_data = list(Data_dict.values())

    layout_list = ["circular", "spring"]
    structure_list_title = ["Watts-Strogatz small world", "Barabasi-Albert scale free"]
    for i, ax in enumerate(axes.flat):
        G = structure_data[i].network
        individual_culture = [x.history_culture[0] for x in structure_data[i].agent_list]#get intial culture
        colour_adjust = norm_zero_one(individual_culture)
        ani_step_colours = cmap(colour_adjust)

        nx.draw(
            G,
            ax=ax,
            node_color=ani_step_colours,
            node_size=50,
            edgecolors="black",
            pos=prod_pos(layout_list[i], G),
        )

        #ax.set_xlabel(r"Time")
        #ax.set_ylabel(r"%s" % y_title)
        ax.set_title(structure_list_title[i])
        #ax.set_ylim(0, 1)

    plotName = fileName + "/Prints"
    f = plotName + "/draw_networks_network_structure_%s" % (
        len(structure_list),
    )
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")




def plot_culture_timeseries(fileName, Data, dpi_save):
    fig, ax = plt.subplots(figsize=(10,6))
    y_title = r"Identity, $I_{t,n}$"

    for v in Data.agent_list:
        ax.plot(np.asarray(Data.history_time), np.asarray(v.history_culture))
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_ylim(0, 1)

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_culture_timeseries"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

"""

def plot_culture_density_timeseries_single(fileName, Data, dpi_save):
    fig, ax = plt.subplots(figsize=(10,6))
    
    y_title = r"Identity, $I_{t,n}$"

    ys_array = np.asarray([v.history_culture for v in Data.agent_list])
    x_array = np.asarray(Data.history_time)

    im = dense_lines(ys=ys_array, x=x_array, ax=ax, cmap='magma')  # this is fast and clean
    cbar = fig.colorbar(im)
    cbar.set_label(r"Density")

    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)
    ax.set_ylim(0, 1)

    #plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_culture_density_timeseries_single.eps"
    fig.savefig(f, dpi=dpi_save, format="eps")

def plot_culture_density_timeseries_multi(fileName, ys_array,x_array, dpi_save):
    fig, ax = plt.subplots(figsize=(10,6))
    
    y_title = r"Identity, $I_{t,n}$"

    im = dense_lines(ys=ys_array, x=x_array, ax=ax, cmap='magma')  # this is fast and clean
    cbar = fig.colorbar(im)
    cbar.set_label(r"Density")

    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)
    ax.set_ylim(0, 1)

    #plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_culture_density_timeseries_multi.eps"
    fig.savefig(f, dpi=dpi_save, format="eps")

def print_culture_density_timeseries_multi(
    fileName, ys_array_list, x_array, title_list, nrows, ncols, dpi_save, ny
):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    y_title = r"Identity, $I_{t,n}$"

    #print(ys_array_list[0].shape)
    for i, ax in enumerate(axes.flat):
        im = dense_lines(ys=ys_array_list[i], x=x_array, ax=ax, cmap='magma', ny = ny)  # this is fast and clean
        cbar = fig.colorbar(im)
        cbar.set_label(r"Density")
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title(title_list[i])

    plt.tight_layout()

    plotName = fileName + "/Prints"
    f = plotName + "/print_culture_density_timeseries_multi.png"
    fig.savefig(f, dpi=dpi_save, format="png")
"""

def plot_compare_av_culture_seed(
    fileName, 
    data_no_culture,
    data_culture, 
    nrows, 
    ncols, 
    dpi_save,
    property_values_list_no_culture, 
    property_varied_no_culture, 
    property_values_list_culture, 
    property_varied_culture
    ):

    y_title = r"Identity, $I_{t,n}$"

    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, sharey=True)

    #print("axes",axes)
    for i in range(len(data_no_culture)):
        # print(np.asarray(Data_list[i].history_average_culture))
        culture_min = np.asarray(data_no_culture[i].history_min_culture)  # bodge
        culture_max = np.asarray(data_no_culture[i].history_max_culture)  # bodge

        axes[0].plot(
            np.asarray(data_no_culture[i].history_time),
            np.asarray(data_no_culture[i].history_average_culture),
            label="%s = %s" % (property_varied_no_culture, property_values_list_no_culture[i]),
        )

        axes[0].fill_between(
            np.asarray(data_no_culture[i].history_time),
            culture_min,
            culture_max,
            alpha=0.5,
            linewidth=0,
        )


    
    for i in range(len(data_culture)):
        # print(np.asarray(Data_list[i].history_average_culture))
        culture_min = np.asarray(data_culture[i].history_min_culture) 
        culture_max = np.asarray(data_culture[i].history_max_culture)  

        axes[1].plot(
            np.asarray(data_culture[i].history_time),
            np.asarray(data_culture[i].history_average_culture),
            label="%s = %s" % (property_varied_culture, property_values_list_culture[i]),
        )

        axes[1].fill_between(
            np.asarray(data_culture[i].history_time),
            culture_min,
            culture_max,
            alpha=0.5,
            linewidth=0,
        )
    
    axes[0].set_ylabel(r"%s" % y_title)
    axes[0].set_xlabel(r"Time")
    axes[0].set_title(r"Behavioural independance")
    axes[0].legend()

    axes[1].set_xlabel(r"Time")
    axes[1].set_title(r"Behavioural dependance")
    axes[1].legend()


    plotName = fileName + "/Prints"

    f = plotName + "/plot_compare_av_culture_seed"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_timeseries_identity_seed(
    fileName, 
    data_no_culture,
    data_culture, 
    nrows, 
    ncols, 
    dpi_save,
    property_values_list_no_culture, 
    property_varied_no_culture, 
    property_values_list_culture, 
    property_varied_culture,
    colour_list
    ):

    y_title = r"Identity, $I_{t,n}$"

    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, sharey=True)

    for i in range(len(data_no_culture)):
        for v in range(data_no_culture[i].N):
            axes[0].plot(
                np.asarray(data_no_culture[i].history_time),
                np.asarray(data_no_culture[i].agent_list[v].history_culture),
                color = colour_list[i]
            )
    
    for i in range(len(data_culture)):
        for v in range(data_culture[i].N):
            axes[1].plot(
                np.asarray(data_culture[i].history_time),
                np.asarray(data_culture[i].agent_list[v].history_culture),
                color = colour_list[i]
            )

    ax.set_ylabel(r"%s" % y_title)
    ax.set_xlabel(r"Time")

    plotName = fileName + "/Prints"

    f = plotName + "/plot_timeseries_identity_seed"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_compare_time_culture_seed(
    fileName, 
    data_no_culture,
    data_culture, 
    nrows, 
    ncols, 
    dpi_save,
    property_values_list_no_culture, 
    property_varied_no_culture, 
    property_values_list_culture, 
    property_varied_culture,
    colour_list
    ):

    y_title = r"Identity, $I_{t,n}$"

    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, sharey=True)

    for i in range(len(data_no_culture)):
        for v in range(data_no_culture[i].N):
            axes[0].plot(
                np.asarray(data_no_culture[i].history_time),
                np.asarray(data_no_culture[i].agent_list[v].history_culture),
                color = colour_list[i]
            )
    
    for i in range(len(data_culture)):
        for v in range(data_culture[i].N):
            axes[1].plot(
                np.asarray(data_culture[i].history_time),
                np.asarray(data_culture[i].agent_list[v].history_culture),
                color = colour_list[i]
            )

    axes[0].set_ylabel(r"%s" % y_title)
    axes[0].set_xlabel(r"Time")
    axes[0].set_title(r"Behavioural independance")

    axes[1].set_xlabel(r"Time")
    axes[1].set_title(r"Behavioural dependance")

    plotName = fileName + "/Prints"

    f = plotName + "/plot_compare_time_culture_seed"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_compare_time_behaviour_culture_seed(
    fileName, 
    data_no_culture,
    data_culture, 
    nrows, 
    ncols, 
    dpi_save,
    colour_list
    ):

    y_title = r"Behavioural Attitude, $A_{t,n,m}$"

    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, sharey=True)
    #print(axes)
    for i in range(len(data_no_culture)):
        for v in range(data_no_culture[i].N):
            data_indivdiual = np.asarray(data_no_culture[i].agent_list[v].history_behaviour_attitudes)
            for j in range(ncols):
                axes[0][j].plot(
                    np.asarray(data_no_culture[i].history_time),
                    data_indivdiual[:,j],
                    color = colour_list[i]
                )

    
    for i in range(len(data_culture)):
        for v in range(data_culture[i].N):
            data_indivdiual = np.asarray(data_culture[i].agent_list[v].history_behaviour_attitudes)
            for j in range(ncols):
                axes[1][j].plot(
                    np.asarray(data_culture[i].history_time),
                    data_indivdiual[:,j],
                    color = colour_list[i]
                )

    fig.supxlabel(r"Time")
    fig.supylabel(r"%s" % y_title)

    plotName = fileName + "/Prints"

    f = plotName + "/plot_compare_time_behaviour_culture_seed"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_network_emissions_timeseries_no_culture(
    fileName, 
    data_no_culture,
    data_culture, 
    dpi_save,
    colour_list
    ):

    y_title = "Step total emissions, $E_{t}$"

    fig, ax = plt.subplots()

    for i in range(len(data_no_culture)):
        ax.plot(
            np.asarray(data_no_culture[i].history_time),
            np.asarray(data_no_culture[i].history_total_carbon_emissions),
            color = colour_list[i],
            linestyle = "dashed"
        )
    
    for i in range(len(data_culture)):
        ax.plot(
            np.asarray(data_culture[i].history_time),
            np.asarray(data_culture[i].history_total_carbon_emissions),
            color = colour_list[i],
            linestyle = "solid"

        )

    ax.set_ylabel(r"%s" % y_title)
    ax.set_xlabel(r"Time")

    plotName = fileName + "/Prints"

    f = plotName + "/plot_network_emissions_timeseries_no_culture"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_behaviorual_emissions_timeseries_no_culture(
    fileName, 
    data_no_culture,
    data_culture, 
    dpi_save,
    colour_list
    ):

    y_title = "Step total behavioural emissions, $E_{t,m}$"

    fig, ax = plt.subplots()

    M = data_no_culture[0].M
    for i in range(len(data_no_culture)):
        data_matrix = np.asarray([[sum(data_no_culture[i].agent_list[v].history_behavioural_carbon_emissions[t][j] for v in range(data_no_culture[i].N)) for t in range(len(data_no_culture[i].history_time))] for j in range(M)])
        for j in range(M):
            ax.plot(
                np.asarray(data_no_culture[i].history_time),
                data_matrix[j],
                color = colour_list[i],
                linestyle = "dashed",
            )
    
    for i in range(len(data_culture)):
        data_matrix = np.asarray([[sum(data_culture[i].agent_list[v].history_behavioural_carbon_emissions[t][j] for v in range(data_culture[i].N)) for t in range(len(data_culture[i].history_time))] for j in range(M)])
        for j in range(M):
            ax.plot(
                np.asarray(data_culture[i].history_time),
                data_matrix[j],
                color = colour_list[i],
                linestyle = "solid",
            )

    ax.set_ylabel(r"%s" % y_title)
    ax.set_xlabel(r"Time")

    plotName = fileName + "/Prints"

    f = plotName + "/plot_behaviorual_emissions_timeseries_no_culture"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def plot_network_timeseries(
    fileName: str, Data: Network, y_title: str, property: str, dpi_save: int
):

    fig, ax = plt.subplots(figsize=(10,6))
    data = eval("Data.%s" % property)

    # bodge
    ax.plot(Data.history_time, data)
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    plotName = fileName + "/Plots"
    f = plotName + "/" + property + "_timeseries.eps"
    fig.savefig(f, dpi=dpi_save, format="eps")


def plot_cultural_range_timeseries(fileName: str, Data, dpi_save: int):
    y_title = "Identity variance"
    property = "history_var_culture"
    plot_network_timeseries(fileName, Data, y_title, property, dpi_save)


def plot_weighting_matrix_convergence_timeseries(
    fileName: str, Data, dpi_save: int
):
    y_title = "Change in Agent Link Strength"
    property = "history_weighting_matrix_convergence"
    plot_network_timeseries(fileName, Data, y_title, property, dpi_save)


def plot_total_carbon_emissions_timeseries(
    fileName: str, Data, dpi_save: int
):
    y_title = "Carbon Emissions"
    property = "history_total_carbon_emissions"
    plot_network_timeseries(fileName, Data, y_title, property, dpi_save)


def plot_green_adoption_timeseries(fileName: str, Data, dpi_save: int):
    y_title = "Green adoption %"
    property = "history_green_adoption"

    plot_network_timeseries(fileName, Data, y_title, property, dpi_save)


def plot_average_culture_timeseries(fileName: str, Data, dpi_save: int):
    y_title = "Average identity"
    property = "history_average_culture"

    plot_network_timeseries(fileName, Data, y_title, property, dpi_save)


def plot_cum_link_change_per_agent(fileName: str, Data: Network, dpi_save: int):

    fig, ax = plt.subplots(figsize=(10,6))
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




def plot_value_timeseries(fileName: str, Data, dpi_save: int):
    y_title = "Behavioural value, B"
    property = "history_behaviour_values"
    ylim_low = -1

    plot_individual_timeseries(fileName, Data, y_title, property, dpi_save, ylim_low)


def plot_attitude_timeseries(fileName: str, Data, dpi_save: int):
    y_title = "Behavioural attiude, A"
    property = "history_behaviour_attitudes"
    ylim_low = 0

    plot_individual_timeseries(fileName, Data, y_title, property, dpi_save, ylim_low)


def plot_threshold_timeseries(fileName: str, Data, dpi_save: int):
    y_title = "Behavioural threshold, T"
    property = "history_behaviour_thresholds"
    ylim_low = 0

    plot_individual_timeseries(fileName, Data, y_title, property, dpi_save, ylim_low)


#################################################################################################################


def live_print_culture_timeseries(
    fileName, Data_list, property_varied, title_list, nrows, ncols, dpi_save
):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(10, 6), sharey=True)#, ,figsize=(14, 7)
    y_title = r"Identity, $I_{t,n}$"

    for i, ax in enumerate(axes.flat):
        for v in Data_list[i].agent_list:
            ax.plot(
                np.asarray(Data_list[i].history_time), np.asarray(v.history_culture)
            )
            #print("v.history_culture",v.history_culture)
        
        ax.text(0.5, 1.03, string.ascii_uppercase[i], transform=ax.transAxes, size=20, weight='bold')

        ax.set_xlabel(r"Time")
        ax.set_ylim(0, 1)

    axes[0].set_ylabel(r"%s" % y_title)

    plt.tight_layout()

    plotName = fileName + "/Prints"
    f = plotName + "/live_plot_culture_timeseries_%s.eps" % property_varied
    fig.savefig(f, dpi=dpi_save, format="eps")


def live_print_culture_timeseries_with_weighting(
    fileName, Data_list, property_varied, title_list, nrows, ncols, dpi_save, cmap
):

    fig, axes = plt.subplots(
        nrows=2, ncols=ncols, figsize=(14, 7), constrained_layout=True
    )
    #print("axes", axes)
    y_title = r"Identity, $I_{t,n}$"

    for i in range(ncols):
        for v in Data_list[i].agent_list:
            axes[0][i].plot(
                np.asarray(Data_list[i].history_time), np.asarray(v.history_culture)
            )

        axes[0][i].set_xlabel(r"Time")
        axes[0][i].set_ylabel(r"%s" % y_title)
        axes[0][i].set_title(title_list[i], pad=5)
        axes[0][i].set_ylim(0, 1)

        axes[1][i].matshow(
            Data_list[i].history_weighting_matrix[-1],
            cmap=cmap,
            norm=Normalize(vmin=0, vmax=1),
            aspect="auto",
        )
        axes[1][i].set_xlabel(r"Individual $k$")
        axes[1][i].set_ylabel(r"Individual $n$")
        # print("matrix", Data_list[i].history_weighting_matrix[-1])
        # print("alpha_change",Data_list[i].alpha_change)
    # colour bar axes
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1)),
        ax=axes[1]#axes.ravel().tolist(),
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label(r"Social network weighting, $\alpha_{n,k}$", labelpad= 5)

    plotName = fileName + "/Prints"
    f = (
        plotName
        + "/lowres_live_print_culture_timeseries_with_weighting_%s.png"
        % property_varied
    )
    fig.savefig(f, dpi=dpi_save, format="png")
    # fig.savefig(f, dpi=dpi_save,format='eps')

def weighting_histogram(
    fileName: str, Data, dpi_save,bin_num
):
    fig, ax = plt.subplots()
    # print("property = ", property)

    triu_weighting = np.triu(Data.history_weighting_matrix[-1])
    flat_data = (triu_weighting).flatten()

    ax.hist(flat_data, density=True, bins = bin_num)  # density=False would make counts
    ax.set_xlabel(r"Social network weighting $\alpha_{n,k}$")
    ax.set_ylabel(r"Count")
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/weighting_histogram.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')




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

    y_title = r"Identity, $I_{t,n}$"

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

    fig, ax = plt.subplots(figsize=(10, 6))

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

def live_varaince_timeseries(
    fileName,
    Data_list,
    property_varied,
    property_varied_title,
    property_varied_values,
    dpi_save,
):

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True
    )

    time_data = Data_list[0].history_time

    for i in range(len(Data_list)):
        y_data =  Data_list[i].history_var_culture
        ax.plot(time_data,y_data, label= property_varied_title + " = " + str(property_varied_values[i]))
    
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Variance, $\sigma^2$")
    ax.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/live_varaince_timeseries_%s_and_%s.eps" % (
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

    fig, ax = plt.subplots(figsize=(10, 6))
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

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)  #
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\sigma /\mu$")

    for i in variable_parameters_dict.keys():
        colour_adjust = variable_parameters_dict[i]["norm"](
            variable_parameters_dict[i]["vals"]
        )
        scatter_colours = get_cmap(variable_parameters_dict[i]["cmap"])(colour_adjust)
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

def linear_Y(
    fileName, combined_data, variable_parameters_dict, dpi_save, y_param, y_title
):
    print("UNFINISHED")
    fig, axes = plt.subplots(
        figsize=(14, 7),
        constrained_layout=True,
        nrows=1,
        ncols=len(variable_parameters_dict.keys()),
    )  #

    key_list = list(variable_parameters_dict.keys())

    for i, ax in enumerate(axes.flat):
        ax.set_xlabel("Normalised parameter")
        ax.set_ylabel(y_title)

        norm_vals = variable_parameters_dict[key_list[i]]["vals"]


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

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

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

def homophily_contour(
    f,total_difference_init_matrix,prob_rewire_list_pos,prob_rewire_list_label,homophily_list_pos,homophily_list_label,cmap, dpi_save, col_label, row_label, y_label,homophily_list,prob_rewire_list
):
    fig, ax = plt.subplots(figsize=(10, 6))#
    
    X, Y = np.meshgrid(prob_rewire_list,homophily_list)
    contours = ax.contour(X, Y, total_difference_init_matrix, colors="black")
    ax.clabel(contours, inline=True, fontsize=8)

    cp = ax.contourf(X, Y, total_difference_init_matrix, cmap=cmap, alpha=0.5)
    cbar = fig.colorbar(
        cp,
        ax=ax,
    )
    cbar.set_label(y_label)

    #ax.set_xticks(prob_rewire_list_pos) 
    #ax.set_xticklabels(prob_rewire_list_label, fontsize=12)
    #ax.xaxis.set_ticks_position('bottom')

    #ax.set_yticks(homophily_list_pos) 
    #ax.set_yticklabels(homophily_list_label, fontsize=12)

    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)

    plotName = f + "/Plots"
    f = plotName + "/Total_difference_between_neighbours_per_agent"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    #fig.savefig(f + ".png", dpi=dpi_save, format="png")


def live_average_multirun_double_phase_diagram_mean_alt(
    fileName, Z, variable_parameters_dict, cmap, dpi_save
):

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

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

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

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

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

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
    fileName, Z, Y_title, Y_param, variable_parameters_dict, cmap, dpi_save, levels
):

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]

    ax.set_xlabel(r"Initial attitude Beta, $b_A$")
    ax.set_ylabel(r"Initial attitude Beta, $a_A$")

    #if col_dict["divisions"] == "log":
    #    ax.set_xscale("log")
    #if row_dict["divisions"] == "log":
    #    ax.set_yscale("log")

    X, Y = np.meshgrid(col_dict["vals"], row_dict["vals"])
    #contours = ax.contour(X, Y, Z, colors="black", levels = levels)
    #ax.clabel(contours, inline=True, fontsize=8)

    cp = ax.contourf(X, Y, Z, cmap=cmap, alpha=0.5, levels = levels)
    cbar = fig.colorbar(
        cp,
        ax=ax,
    )
    cbar.set_label(Y_title)

    plotName = fileName + "/Plots"
    f = plotName + "/live_average_multirun_double_phase_diagram_%s" % (Y_param)
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def double_phase_diagram_using_meanandvariance(
    fileName, Z, Y_title, Y_param, variable_parameters_dict, cmap, dpi_save
):

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

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
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def double_matrix_plot(
    fileName, Z, Y_title, Y_param, variable_parameters_dict, cmap, dpi_save,x_ticks_pos,y_ticks_pos,x_ticks_label,y_ticks_label
):

    fig, ax = plt.subplots()#figsize=(10, 6)

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]

    #ax.set_xlabel(col_dict["title"])
    #ax.set_ylabel(row_dict["title"])

    ax.set_xlabel(r"Initial attitude Beta, $a_A$")
    ax.set_ylabel(r"Initial attitude Beta, $b_A$")

    X, Y = np.meshgrid(col_dict["vals"], row_dict["vals"])

    

    mat = ax.matshow(
        Z,
        cmap=cmap,
        aspect="auto",
    )
    cbar = fig.colorbar(
        mat,
        #plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=Z.min(), vmax=Z.max())),
        ax=ax,
    )
    cbar.set_label(Y_title)

    if col_dict["divisions"] == "log":
        ax.set_xscale("log")
    if row_dict["divisions"] == "log":
        ax.set_yscale("log")


    # HAS TO BE AFTER PUTTING INT THE MATRIX 
    #ax.set_xticks(x_ticks_pos)
    #ax.set_xticklabels(x_ticks_label)  

    ax.xaxis.set_ticks_position('bottom')

    #ax.set_yticks(y_ticks_pos)
    ##ax.set_yticklabels(y_ticks_label)  

    plotName = fileName + "/Plots"
    f = plotName + "/live_average_double_matrix_plot_%s" % (Y_param)
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    #fig.savefig(f + ".png", dpi=dpi_save, format="png")

def double_matrix_plot_cluster(
    fileName, Z, variable_parameters_dict, cmap, dpi_save, col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label
):

    fig, ax = plt.subplots(figsize=(10, 6))

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]

    ax.set_xlabel(r"Confirmation bias, $\theta$")
    ax.set_ylabel(r"Attitude Beta parameters $a, 2-b$")

    if col_dict["divisions"] == "log":
        ax.set_xscale("log")
    if row_dict["divisions"] == "log":
        ax.set_yscale("log")

    mat = ax.matshow(
        Z,
        cmap=cmap,
        aspect="auto",
    )
    cbar = fig.colorbar(
        mat,
        #plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=Z.min(), vmax=Z.max())),
        ax=ax,
    )
    cbar.set_label(r"Mean number of identity bubbles")

    # HAS TO BE AFTER PUTTING INT THE MATRIX 
    ax.set_xticks(col_ticks_pos)
    ax.set_xticklabels(col_ticks_label)  
    ax.xaxis.set_ticks_position('bottom')

    ax.set_yticks(row_ticks_pos)
    ax.set_yticklabels(row_ticks_label)  

    plotName = fileName + "/Plots"
    f = plotName + "/live_average_double_matrix_plot_cluster_count"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def double_contour_plot_cluster(
    fileName, Z, variable_parameters_dict, cmap, dpi_save, col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label,property_varied_values_col, property_varied_values_row
):

    fig, ax = plt.subplots(figsize=(10, 6))

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]

    ax.set_xlabel(r"Confirmation bias, $\theta$")
    ax.set_ylabel(r"Attitude Beta parameters $a, 2-b$")

    if col_dict["divisions"] == "log":
        ax.set_xscale("log")
    if row_dict["divisions"] == "log":
        ax.set_yscale("log")

    X, Y = np.meshgrid(property_varied_values_col, property_varied_values_row)
    contours = ax.contour(X, Y, Z, colors="black")
    #ax.clabel(contours, inline=True, fontsize=8)

    cp = ax.contourf(X, Y, Z, cmap=cmap, alpha=0.5)

    cbar = fig.colorbar(
        cp,
        #plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=Z.min(), vmax=Z.max())),
        ax=ax,
    )
    cbar.set_label(r"Mean number of identity bubbles")

    # HAS TO BE AFTER PUTTING INT THE MATRIX 
    #ax.set_xticks(col_ticks_pos)
    #ax.set_xticklabels(col_ticks_label)  
    ax.xaxis.set_ticks_position('bottom')

    #ax.set_yticks(row_ticks_pos)
    #ax.set_yticklabels(row_ticks_label)  

    plotName = fileName + "/Plots"
    f = plotName + "/double_contour_plot_cluster"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def double_matrix_plot_cluster_multi(
    fileName, Z_list, variable_parameters_dict, cmap, dpi_save, col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label, nrows, ncols, bandwidth_list
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(10, 6))

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]

    for i, ax in enumerate(axes.flat):

        if col_dict["divisions"] == "log":
            ax.set_xscale("log")
        if row_dict["divisions"] == "log":
            ax.set_yscale("log")

        mat = ax.matshow(
            Z_list[i],
            cmap=cmap,
            aspect="auto",
        )


        
        #ax.set_xlabel(r"Confirmation bias, $\theta$")
        #ax.set_ylabel(r"Attitude Beta parameters $a, 2-b$")

        # HAS TO BE AFTER PUTTING INT THE MATRIX 
        ax.set_xticks(col_ticks_pos)
        ax.set_xticklabels(col_ticks_label)  
        ax.xaxis.set_ticks_position('bottom')

        ax.set_yticks(row_ticks_pos)
        ax.set_yticklabels(row_ticks_label)  

        ax.set_title(r"Bandwidth = %s" % bandwidth_list[i])

    #print("mins", [np.min(Z) for Z in Z_list])
    #print("mazx", [np.max(Z) for Z in Z_list])

    Z_min = min([np.min(Z) for Z in Z_list])
    Z_max = max([np.max(Z) for Z in Z_list])

    #print("Z_min",Z_min)
    #print("Z_max",Z_max)

    cbar = fig.colorbar(
        #plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1)),
        plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=Z_min, vmax=Z_max)),
        ax=axes.ravel().tolist(),
    )
    cbar.set_label(r"Mean number of identity bubbles")

    fig.supxlabel(r"Confirmation bias, $\theta$")
    fig.supylabel(r"Attitude Beta parameters $a/b$")

    plotName = fileName + "/Plots"
    f = plotName + "/double_matrix_plot_cluster_multi_%s" %(len(bandwidth_list))
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def double_matrix_plot_cluster_var_multi(
    fileName, Z_list, variable_parameters_dict, cmap, dpi_save, col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label, nrows, ncols, bandwidth_list
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(10, 6))

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]

    for i, ax in enumerate(axes.flat):

        if col_dict["divisions"] == "log":
            ax.set_xscale("log")
        if row_dict["divisions"] == "log":
            ax.set_yscale("log")

        mat = ax.matshow(
            Z_list[i],
            cmap=cmap,
            aspect="auto",
        )


        
        #ax.set_xlabel(r"Confirmation bias, $\theta$")
        #ax.set_ylabel(r"Attitude Beta parameters $a, 2-b$")

        # HAS TO BE AFTER PUTTING INT THE MATRIX 
        ax.set_xticks(col_ticks_pos)
        ax.set_xticklabels(col_ticks_label)  
        ax.xaxis.set_ticks_position('bottom')

        ax.set_yticks(row_ticks_pos)
        ax.set_yticklabels(row_ticks_label)
        ax.set_title(r"Bandwidth = %s" % bandwidth_list[i])

    #print("mins", [np.min(Z) for Z in Z_list])
    #print("mazx", [np.max(Z) for Z in Z_list])

    Z_min = min([np.min(Z) for Z in Z_list])
    Z_max = max([np.max(Z) for Z in Z_list])

    #print("Z_min",Z_min)
    #print("Z_max",Z_max)

    cbar = fig.colorbar(
        #plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1)),
        plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=Z_min, vmax=Z_max)),
        ax=axes.ravel().tolist(),
    )
    cbar.set_label(r"Variance in number of identity bubbles, $\sigma^2_{C}$")

    fig.supxlabel(r"Confirmation bias, $\theta$")
    fig.supylabel(r"Attitude Beta parameters $a/b$")

    plotName = fileName + "/Plots"
    f = plotName + "/double_matrix_plot_cluster_multi_%s" %(len(bandwidth_list))
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def double_matrix_plot_cluster_ratio(
        fileName, Z, variable_parameters_dict, cmap, dpi_save, col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label
):

    fig, ax = plt.subplots(figsize=(10, 6))

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]

    ax.set_xlabel(r"Confirmation bias, $\theta$")
    ax.set_ylabel(r"Attitude Beta parameters $a/b$")

    if col_dict["divisions"] == "log":
        ax.set_xscale("log")
    if row_dict["divisions"] == "log":
        ax.set_yscale("log")

    mat = ax.matshow(
        Z,
        cmap=cmap,
        aspect="auto",
    )
    cbar = fig.colorbar(
        mat,
        #plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=Z.min(), vmax=Z.max())),
        ax=ax,
    )
    cbar.set_label(r"Number of identity bubbles")

    # HAS TO BE AFTER PUTTING INT THE MATRIX 
    ax.set_xticks(col_ticks_pos)
    ax.set_xticklabels(col_ticks_label)  
    ax.xaxis.set_ticks_position('bottom')

    ax.set_yticks(row_ticks_pos)
    ax.set_yticklabels(row_ticks_label)  

    plotName = fileName + "/Plots"
    f = plotName + "/ratio_live_average_double_matrix_plot_cluster_count_ratio"
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def double_matrix_plot_ab(fileName, Z, Y_title, Y_param, variable_parameters_dict, cmap, dpi_save,col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label
):

    fig, ax = plt.subplots()#figsize=(10, 6)

    col_dict = variable_parameters_dict["col"]
    row_dict = variable_parameters_dict["row"]

    mat = ax.matshow(
        Z,
        cmap=cmap,
        aspect="auto",
    )
    cbar = fig.colorbar(
        mat,
        #plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=Z.min(), vmax=Z.max())),
        ax=ax,
    )
    cbar.set_label(Y_title)
    ax.xaxis.set_ticks_position('bottom')
    
    ax.set_xlabel(r"Initial attitude Beta, $a_A$")
    ax.set_ylabel(r"Initial attitude Beta, $b_A$")

    if col_dict["divisions"] == "log":
        ax.set_xscale("log")
        print("LOOGGGG")
    if row_dict["divisions"] == "log":
        ax.set_yscale("log")

    #ax.set_xticks(col_ticks_pos)
    #ax.set_xticklabels(col_ticks_label)  
    ax.xaxis.set_ticks_position('bottom')

    #ax.set_yticks(row_ticks_pos)
    #ax.set_yticklabels(row_ticks_label)  
    #print("Y_param",Y_param)

    plotName = fileName + "/Plots"
    f = plotName + "/live_average_double_matrix_plot_%s" % (Y_param)
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    #fig.savefig(f + ".png", dpi=dpi_save, format="png")

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------

    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def homophily_matrix(f,total_difference_init_matrix,prob_rewire_list_pos,prob_rewire_list_label,homophily_list_pos,homophily_list_label,cmap, dpi_save, col_label, row_label, y_label):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mat = ax.matshow(
        total_difference_init_matrix,
        aspect="auto",
        cmap=cmap,
    )
    cbar = fig.colorbar(
        mat,
        #plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=Z.min(), vmax=Z.max())),
        ax=ax,
    )
    cbar.set_label(y_label)

    ax.set_xticks(prob_rewire_list_pos) 
    ax.set_xticklabels(prob_rewire_list_label, fontsize=12)
    ax.xaxis.set_ticks_position('bottom')

    ax.set_yticks(homophily_list_pos) 
    ax.set_yticklabels(homophily_list_label, fontsize=12)

    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)

    plotName = f + "/Plots"
    f = plotName + "/Total_difference_between_neighbours_per_agent"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    #fig.savefig(f + ".png", dpi=dpi_save, format="png")

def multi_line_matrix_plot(
    fileName, Z, col_vals, row_vals,  Y_param, cmap, dpi_save, col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label,col_axis_x, col_label, row_label, y_label
    ):

    fig, ax = plt.subplots( constrained_layout=True)#figsize=(14, 7)

    if col_axis_x:
        xs = np.tile(col_vals, (len(row_vals), 1))
        ys = Z
        c = row_vals

    else:
        xs = np.tile(row_vals, (len(col_vals), 1))
        ys = np.transpose(Z)
        c = col_vals
    
    #print("after",xs.shape, ys.shape, c.shape)

    ax.set_ylabel(y_label)#(r"First behaviour attitude variance, $\sigma^2$")
    
    #plt.xticks(x_ticks_pos, x_ticks_label)

    lc = multiline(xs, ys, c, cmap=cmap, lw=2)
    axcb = fig.colorbar(lc)

    if col_axis_x:
        axcb.set_label(row_label)#(r"Number of behaviours per agent, M")
        ax.set_xlabel(col_label)#(r'Confirmation bias, $\theta$')
        ax.set_xticks(col_ticks_pos)
        ax.set_xticklabels(col_ticks_label)
        print("x ticks", col_label,col_ticks_pos, col_ticks_label)
        #ax.set_xlim(left = 0.0, right = 60)
        #ax.set_xlim(left = -10.0, right = 90)

    else:
        axcb.set_label(col_label)#)(r'Confirmation bias, $\theta$')
        ax.set_xlabel(row_label)#(r"Number of behaviours per agent, M")
        ax.set_xticks(row_ticks_pos)
        ax.set_xticklabels(row_ticks_label)
        print("x ticks", row_label, row_ticks_pos, row_ticks_label)
        #ax.set_xlim(left = 1.0)
        #ax.set_xlim(left = 0.0, right = 2.0)

    plotName = fileName + "/Plots"
    f = plotName + "/multi_line_matrix_plot_%s_%s" % (Y_param, col_axis_x)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def multi_line_matrix_plot_divide_through(
    fileName, Z, col_vals, row_vals,  Y_param, cmap, dpi_save, col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label,col_axis_x
    ):

    fig, ax = plt.subplots( figsize=(14, 7),constrained_layout=True)#figsize=(14, 7)

    if col_axis_x:
        xs = np.tile(col_vals, (len(row_vals), 1))
        ys = Z
        c = row_vals
    else:
        xs = np.tile(row_vals, (len(col_vals), 1))
        ys = np.transpose(Z)
        c = col_vals

    lc = multiline(xs, ys, c, cmap=cmap, lw=2)#returns lien collection
    axcb = fig.colorbar(lc)

    ax.set_ylim(bottom = 0.0)
    ax.set_ylabel(r"Normalised behavioural attitude variance, $\sigma^{2}_{2}/\sigma^{2}_{1}$")

    if col_axis_x:
        axcb.set_label(r"Number of behaviours per agent, $M$")
        ax.set_xlabel(r'Confirmation bias, $\theta$')
        ax.set_xticks(col_ticks_pos)
        ax.set_xticklabels(col_ticks_label)
        ax.set_xlim(left = 0.0, right = 60)
    else:
        axcb.set_label(r'Confirmation bias, $\theta$')
        ax.set_xlabel(r"Number of behaviours per agent, $M$")
        ax.set_xticks(row_ticks_pos)
        ax.set_xticklabels(row_ticks_label)
        ax.set_xlim(left = 1.0)

    plotName = fileName + "/Plots"
    f = plotName + "/multi_line_matrix_plot_divide_through_%s_%s" % (Y_param, col_axis_x)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")


def multi_line_matrix_plot_difference(
    fileName, Z, col_vals, row_vals,  Y_param, cmap, dpi_save, col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label,col_axis_x, col_label, row_label, y_label
    ):

    fig, ax = plt.subplots( constrained_layout=True)#figsize=(14, 7)

    if col_axis_x:
        xs = np.tile(col_vals, (len(row_vals), 1))
        ys = Z
        c = row_vals

    else:
        xs = np.tile(row_vals, (len(col_vals), 1))
        ys = np.transpose(Z)
        c = col_vals
    
    #print("after",xs.shape, ys.shape, c.shape)

    ax.set_ylabel(y_label)#(r"First behaviour attitude variance, $\sigma^2$")
    
    #plt.xticks(x_ticks_pos, x_ticks_label)

    lc = multiline(xs, ys, c, cmap=cmap, lw=2)
    axcb = fig.colorbar(lc)

    if col_axis_x:
        axcb.set_label(row_label)#(r"Number of behaviours per agent, M")
        ax.set_xlabel(col_label)#(r'Confirmation bias, $\theta$')
        ax.set_xticks(col_ticks_pos)
        ax.set_xticklabels(col_ticks_label)
        print("x ticks", col_label,col_ticks_pos, col_ticks_label)
        #ax.set_xlim(left = 0.0, right = 60)
        #ax.set_xlim(left = -10.0, right = 90)

    else:
        axcb.set_label(col_label)#)(r'Confirmation bias, $\theta$')
        ax.set_xlabel(row_label)#(r"Number of behaviours per agent, M")
        ax.set_xticks(row_ticks_pos)
        ax.set_xticklabels(row_ticks_label)
        print("x ticks", row_label, row_ticks_pos, row_ticks_label)
        #ax.set_xlim(left = 1.0)
        #ax.set_xlim(left = 0.0, right = 2.0)

    plotName = fileName + "/Plots"
    f = plotName + "/multi_line_matrix_plot_difference_%s_%s" % (Y_param, col_axis_x)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def multi_line_matrix_plot_difference_percentage(
    fileName, Z, col_vals, row_vals,  Y_param, cmap, dpi_save, col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label,col_axis_x, col_label, row_label, y_label
    ):

    fig, ax = plt.subplots( constrained_layout=True)#figsize=(14, 7)

    if col_axis_x:
        xs = np.tile(col_vals, (len(row_vals), 1))
        ys = Z
        c = row_vals

    else:
        xs = np.tile(row_vals, (len(col_vals), 1))
        ys = np.transpose(Z)
        c = col_vals
    
    #print("after",xs.shape, ys.shape, c.shape)

    ax.set_ylabel(y_label)#(r"First behaviour attitude variance, $\sigma^2$")
    
    #plt.xticks(x_ticks_pos, x_ticks_label)

    lc = multiline(xs, ys, c, cmap=cmap, lw=2)
    axcb = fig.colorbar(lc)

    if col_axis_x:
        axcb.set_label(row_label)#(r"Number of behaviours per agent, M")
        ax.set_xlabel(col_label)#(r'Confirmation bias, $\theta$')
        ax.set_xticks(col_ticks_pos)
        ax.set_xticklabels(col_ticks_label)
        #print("x ticks", col_label,col_ticks_pos, col_ticks_label)
        #ax.set_xlim(left = 0.0, right = 60)
        #ax.set_xlim(left = -10.0, right = 90)

    else:
        axcb.set_label(col_label)#)(r'Confirmation bias, $\theta$')
        ax.set_xlabel(row_label)#(r"Number of behaviours per agent, M")
        ax.set_xticks(row_ticks_pos)
        ax.set_xticklabels(row_ticks_label)
        #print("x ticks", row_label, row_ticks_pos, row_ticks_label)
        #ax.set_xlim(left = 1.0)
        #ax.set_xlim(left = 0.0, right = 2.0)

    plotName = fileName + "/Plots"
    f = plotName + "/multi_line_matrix_plot_difference_percentage_%s_%s" % (Y_param, col_axis_x)
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

    y_title = r"Identity, $I_{t,n}$"

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
    fileName: str,
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

    plotName = fileName + "/Prints"
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

    fig, ax = plt.subplots(figsize=(10,6))
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

    fig, ax = plt.subplots(figsize=(10,6))
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
    y_title = "Step total emissions, r$E_{t}$"

    fig, ax = plt.subplots(figsize=(10,6))
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

    fig, ax = plt.subplots(figsize=(10,6))
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

    fig, ax = plt.subplots(figsize=(10,6))
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

    fig, ax = plt.subplots(figsize=(10,6))
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

    fig, ax = plt.subplots(figsize=(10,6))
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

    fig, ax = plt.subplots(figsize=(10,6))
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

    fig, ax = plt.subplots(figsize=(10,6))
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

def print_live_initial_culture_network(
    fileName: str,
    Data,
    dpi_save: int,
    layout: str,
    norm_zero_one,
    cmap,
    node_size,
    round_dec,
):

    fig, ax = plt.subplots()


    G = nx.from_numpy_matrix(Data.history_weighting_matrix[0])
    pos_culture_network = prod_pos(layout, G)

    indiv_culutre_list = [v.history_culture[0] for v in Data.agent_list]
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
        plt.cm.ScalarMappable(cmap=cmap, norm=norm_zero_one), ax=ax
    )
    cbar.set_label(r"Identity, $I_{t,n}$")

    plotName = fileName + "/Prints"
    f = plotName + "/print_live_intial_culture_network"
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

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
    cbar.set_label(r"Identity, $I_{t,n}$")

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
    print("HEY!",layout)
    y_title = r"Identity, $I_{t,n}$"
    fig, axes = plt.subplots(
        nrows=2, ncols=ncols, figsize=(14, 7), constrained_layout=True
    )

    for i in range(ncols):
        #####NETWORK
        G = nx.from_numpy_matrix(Data_list[i].history_weighting_matrix[0])
        
        pos_culture_network = prod_pos(layout[i], G)
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
        axes[0][i].set_ylabel(r"%s" % y_title, labelpad=5)
        axes[0][i].set_ylim(0, 1)

    # colour bar axes
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm_zero_one), ax=axes[1]
    )
    cbar.set_label(r"Initial identity, $I_{0,n}$")

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
    fileName: str,
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
    cbar_culture.set_label(r"Identity, $I_{t,n}$")

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
    animateName = fileName + "/Animations"
    f = (
        animateName
        + "/live_multi_animate_culture_network_and_weighting_%s.mp4" % property_name
    )
    # print("f", f)
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)

    return ani

# animation of changing culture
def live_animate_weighting_matrix(
    fileName: str,
    Data: list,
    cmap_weighting: Union[LinearSegmentedColormap, str],
    interval: int,
    fps: int,
    round_dec: int,
):
    def update(i, Data, ax, title):

        ax.clear()

        ax.matshow(
            Data.history_weighting_matrix[i],
            cmap=cmap_weighting,
            norm=Normalize(vmin=0, vmax=1),
            aspect="auto",
        )

        ax.set_xlabel("Individual $k$")
        ax.set_ylabel("Individual $n$")

        title.set_text(
            "Time= {}".format(round(Data.history_time[i], round_dec))
        )

    fig, ax = plt.subplots()

    # plt.tight_layout()

    title = plt.suptitle(t="", fontsize=20)

    cbar_weight = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_weighting),
        ax=ax,
        location="right",
    )  # This does a mapabble on the fly i think, not sure
    cbar_weight.set_label(r"Social network weighting, $\alpha_{n,k}$")

    # need to generate the network from the matrix
    # G = nx.from_numpy_matrix(Data_list[0].history_weighting_matrix[0])

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(len(Data.history_time)),
        fargs=(Data, ax, title),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = fileName + "/Animations"
    f = animateName + "/live_animate_weighting_matrix.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)

    return ani


# animation of changing culture
def live_compare_animate_weighting_matrix(
    fileName: str,
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
    animateName = fileName + "/Animations"
    f = animateName + "/live_compare_animate_weighting_matrix_%s.mp4" % property_name
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)

    return ani


# animation of changing culture
def live_compare_animate_behaviour_matrix(
    fileName: str,
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
    animateName = fileName + "/Animations"
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
    cbar.set_label(r"Identity, $I_{t,n}$")

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


def bar_sensitivity_analysis_plot(fileName, data, names, yerr, dpi_save, N_samples):
    """
    Create bar chart of results.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    data.plot(kind="barh", xerr=yerr, ax=ax)  # Pandas data frame plot
    ax.set_yticks(
        ticks=range(len(names)), labels=names
    )  # ,rotation=0.0, horizontalalignment="center"
    ax.set_xlim(left=0)
    plt.tight_layout()

    plotName = fileName + "/Prints"
    f = (
        plotName
        + "/"
        + "%s_%s_bar_sensitivity_analysis_plot.eps" % (len(names), N_samples)
    )
    fig.savefig(f, dpi=dpi_save, format="eps")


def scatter_total_sensitivity_analysis_plot(
    fileName, data, names, xerr, dpi_save, N_samples
):
    """
    Create scatter chart of results.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        data["ST"].tolist(), names, xerr=xerr["ST"].tolist(), fmt="o", ecolor="k"
    )

    ax.set_xlim(left=0)
    ax.set_xlabel(r"Total Sobol sensitivity")
    ax.set_ylabel(r"Exogenous parameters")
    plt.tight_layout()

    plotName = fileName + "/Prints"
    f = (
        plotName
        + "/"
        + "%s_%s_scatter_sensitivity_analysis_plot.eps" % (len(names), N_samples)
    )
    fig.savefig(f, dpi=dpi_save, format="eps")


def multi_scatter_total_sensitivity_analysis_plot(
    fileName, data_dict, names, dpi_save, N_samples, order
):
    """
    Create scatter chart of results.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

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

    plotName = fileName + "/Prints"
    f = (
        plotName
        + "/"
        + "%s_%s_%s_multi_scatter_sensitivity_analysis_plot.eps"
        % (len(names), N_samples, order)
    )
    fig.savefig(f, dpi=dpi_save, format="eps")


def multi_scatter_seperate_total_sensitivity_analysis_plot(
    fileName, data_dict, names, dpi_save, N_samples, order
):
    """
    Create scatter chart of results.
    """

    #dict_list = ['var','emissions']#
    #dict_list = ['var','emissions',"emissions_change"]#list(data_dict.keys())
    #dict_list = list(data_dict.keys())
    dict_list = ['emissions','var',"emissions_change"]#"coefficient_of_variance",

    fig, axes = plt.subplots(ncols=len(dict_list), nrows=1, constrained_layout=True , sharey=True,figsize=(12, 6))#,#sharex=True# figsize=(14, 7) # len(list(data_dict.keys())))
    
    plt.rc('ytick', labelsize=4) 

    for i, ax in enumerate(axes.flat):
        if order == "First":
            ax.errorbar(
                data_dict[dict_list[i]]["data"]["S1"].tolist(),
                names,
                xerr=data_dict[dict_list[i]]["yerr"]["S1"].tolist(),
                fmt="o",
                ecolor="k",
                color=data_dict[dict_list[i]]["colour"],
                label=data_dict[dict_list[i]]["title"],
            )
        else:
            ax.errorbar(
                data_dict[dict_list[i]]["data"]["ST"].tolist(),
                names,
                xerr=data_dict[dict_list[i]]["yerr"]["ST"].tolist(),
                fmt="o",
                ecolor="k",
                color=data_dict[dict_list[i]]["colour"],
                label=data_dict[dict_list[i]]["title"],
            )
        ax.legend()
        ax.set_xlim(left=0)
        #ax.set_xlabel(r"%s Sobol sensitivity" % (order))
    #fig.text(0.5, 0.04, r"%s Sobol sensitivity" % (order), ha='center')
    fig.supxlabel(r"%s order Sobol index" % (order))
    #plt.xlabel(r"%s Sobol sensitivity" % (order))
        #ax.set_ylabel(r"Exogenous parameters")
            
        #ax.set_yticklabels(names, rotation = 45)

    plt.tight_layout()

    plotName = fileName + "/Prints"
    f = (
        plotName
        + "/"
        + "%s_%s_%s_multi_scatter_seperate_sensitivity_analysis_plot.eps"
        % (len(names), N_samples, order)
    )
    f_png = (
        plotName
        + "/"
        + "%s_%s_%s_multi_scatter_seperate_sensitivity_analysis_plot.png"
        % (len(names), N_samples, order)
    )
    fig.savefig(f, dpi=dpi_save, format="eps")
    fig.savefig(f_png, dpi=dpi_save, format="png")

def multi_scatter_parallel_total_sensitivity_analysis_plot(
    fileName, data_dict, names, dpi_save, N_samples, order
):
    """
    Create scatter chart of results.
    """
    #print("data",data_dict)
    dict_list = ['var','emissions',"emissions_change"]#list(data_dict.keys())

    fig, ax = plt.subplots(figsize=(13, 6))#sharex=True# figsize=(14, 7) # len(list(data_dict.keys())))

    plt.rc('ytick', labelsize=4) 

    name_pos = np.arange(len(names))
    separation = 0.15
    name_pos_list = [name_pos - separation, name_pos,name_pos + separation]#dependant o the nuber of things FIX

    for i in range(len(dict_list)):
        if order == "First":
            ax.errorbar(
                x=data_dict[dict_list[i]]["data"]["S1"].tolist(),
                y = name_pos_list[i],
                xerr=data_dict[dict_list[i]]["yerr"]["S1"].tolist(),
                fmt="o",
                ecolor="k",
                color=data_dict[dict_list[i]]["colour"],
                label=data_dict[dict_list[i]]["title"],
            )
        else:
            ax.errorbar(
                x=data_dict[dict_list[i]]["data"]["ST"].tolist(),
                y = name_pos_list[i],
                xerr=data_dict[dict_list[i]]["yerr"]["ST"].tolist(),
                fmt="o",
                ecolor="k",
                color=data_dict[dict_list[i]]["colour"],
                label=data_dict[dict_list[i]]["title"],
            )

    ax.legend()
    ax.set_xlim(left=0)
    ax.set_xlabel(r"%s order Sobol sensitivity" % (order))
    ax.set_yticks(ticks=name_pos, labels=names) 
        #ax.set_ylabel(r"Exogenous parameters")
            
        #ax.set_yticklabels(names, rotation = 45)

    plt.tight_layout()

    plotName = fileName + "/Prints"
    f = (
        plotName
        + "/"
        + "%s_%s_%s_multi_scatter_parallel_sensitivity_analysis_plot.eps"
        % (len(names), N_samples, order)
    )
    f_png = (
        plotName
        + "/"
        + "%s_%s_%s_multi_scatter_parallel_sensitivity_analysis_plot.png"
        % (len(names), N_samples, order)
    )
    fig.savefig(f, dpi=dpi_save, format="eps")
    #fig.savefig(f_png, dpi=dpi_save, format="png")


def multi_scatter_sidebyside_total_sensitivity_analysis_plot(
    fileName, data_dict, names, dpi_save, N_samples, order
):
    """
    Create scatter chart of results.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

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

    plotName = fileName + "/Prints"
    f = (
        plotName
        + "/"
        + "%s_%s_%s_multi_scatter_sidebyside_total_sensitivity_analysis_plot.eps"
        % (len(names), N_samples, order)
    )
    fig.savefig(f, dpi=dpi_save, format="eps")


def prints_SA_matrix(
    fileName, Data, title_list, cmap, nrows, ncols, dpi_save, labels, title_property, Y_property
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

    plotName = fileName + "/Prints"
    #f = plotName + "/" + "%s_prints_SA_matrix_property_%s.eps" % (len(labels), Y_property)
    f_png = plotName + "/" + "%s_prints_SA_matrix_property_%s.png" % (len(labels), Y_property)
    #fig.savefig(f, dpi=dpi_save, format="eps")
    fig.savefig(f_png, dpi=dpi_save, format="png")


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

    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(attitudes_list[0][-1], attitudes_list[1][-1])
    ax.set_xlabel(r"Attitude")
    ax.set_ylabel(r"Attitude")

    plotName = fileName + "/Plots"
    f = plotName + "/live_plot_attitude_scatter.eps"
    fig.savefig(f, dpi=dpi_save, format="eps")


def plot_alpha_variation(fileName, num_counts, phi_list, dpi_save):
    def alpha_calc(phi, x):
        return np.exp(-phi * np.abs(x))

    def alpha_diff_calc(phi, x):
        return -phi * np.exp(-phi * np.abs(x))

    fig, ax = plt.subplots(figsize=(10,6))

    x = np.linspace(0, 1, num_counts)

    for i in phi_list:
        y = [alpha_calc(i, x) for x in x]
        ax.plot(x, y, "-", label="Phi = %s" % i)
        dydx = [alpha_diff_calc(i, x) for x in x]
        ax.plot(x, dydx, "--", label="Phi = %s" % i)

    ax.set_xlabel(r"$|I_n -I_k|$")
    ax.set_ylabel(r"$\alpha$")
    ax.legend()
    plotName = fileName + "/Plots"
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
    
    fig, ax = plt.subplots(figsize=(10,6))

    for j in range(int(Data["N"])):
        ax.scatter(PropertyData[0][j][-1], PropertyData[1][j][-1])

    ax.set_xlabel(r"Attitude")
    ax.set_ylabel(r"Attitude")

    plotName = fileName + "/Plots"
    f = plotName + "/plot_attitude_scatter.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')

def plot_weighting_link_timeseries(fileName: str, Data, y_title:str, dpi_save:int, min_val):

    fig, ax = plt.subplots(figsize=(10,6))

    for i in range(int(Data["N"])):
        for v in range(int(Data["N"])):
            if Data["network_weighting_matrix"][0][i][v] > 0.0:
                link_data = [Data["network_weighting_matrix"][x][i][v] for x in range(len(Data["network_time"]))]
                if any(j > min_val for j in link_data):
                    ax.plot(Data["network_time"], link_data)

    ax.axvline(Data["culture_momentum_real"], color='r',linestyle = "--")
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    plotName = fileName + "/Plots"
    f = plotName + "/plot_weighting_link_timeseries.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')

def animate_culture_network_and_weighting(
    fileName: str, Data, layout:str, cmap_culture: Union[LinearSegmentedColormap,str], node_size:int, interval:int, fps:int, norm_zero_one: SymLogNorm, round_dec:int, cmap_edge
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
    fig, ax = plt.subplots(figsize=(10,6))
    # cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax)#This does a mapabble on the fly i think, not sure
    cbar_culture = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax
    )  # This does a mapabble on the fly i think, not sure
    cbar_culture.set_label(r"Identity, $I_{t,n}$")

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
    animateName = fileName + "/Animations"
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

    fig, ax = plt.subplots(figsize=(10,6))
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label(r"Identity, $I_{t,n}$")

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
    fileName: str, Data, property:str, nrows:int, ncols:int, frames_list, round_dec, dpi_save,bin_num
):
    y_title = "Probability"
    #print(Data[property], Data[property].shape)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    # print("property = ", property)

    for i, ax in enumerate(axes.flat):
        #print(Data[property][frames_list[i]])
        ax.hist(Data[property][frames_list[i]], density=True, bins = bin_num)  # density=False would make counts
        ax.set_xlabel(rr"Identity, $I_{t,n}$")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title("Time= {}".format(round(Data["network_time"][frames_list[i]], round_dec)))  # avoid 0 in the title
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/print_culture_histogram.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')

    # make matrix animation
def animate_weighting_matrix(fileName: str, Data, interval:int, fps:int, round_dec:int, cmap_weighting: Union[LinearSegmentedColormap,str]):
    def update(i):
        M = Data["network_weighting_matrix"][i]
        # print("next frame!",M)        
        matrice.set_array(M)
        # Set the title
        
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][i], round_dec))
            )
        return matrice

    fig, ax = plt.subplots(figsize=(10,6))
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
    animateName = fileName + "/Animations"
    f = animateName + "/" + "weighting_matrix_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)
    return ani

# make behaviour evolution plot
def animate_behavioural_matrix(
    fileName: str, Data, interval:int, fps:int, cmap_behaviour: Union[LinearSegmentedColormap,str], round_dec:int
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

    fig, ax = plt.subplots(figsize=(10,6))
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
    animateName = fileName + "/Animations"
    f = animateName + "/" + "behavioural_matrix_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)
    return ani

# animation of changing culture
def animate_culture_network(
    fileName: str, Data, layout:str, cmap_culture: Union[LinearSegmentedColormap,str], node_size:int, interval:int, fps:int, norm_zero_one: SymLogNorm, round_dec:int
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
    fig, ax = plt.subplots(figsize=(10,6))
    # cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax)#This does a mapabble on the fly i think, not sure
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label(r"Identity, $I_{t,n}$")

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
    animateName = fileName + "/Animations"
    f = animateName + "/" + "cultural_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)
    return ani


def prints_behavioural_matrix(
    fileName: str, Data, cmap_behaviour: Union[LinearSegmentedColormap,str], nrows:int, ncols:int, frames_list:list[int], round_dec:int, dpi_save:int
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

    plotName = fileName + "/Prints"
    f = plotName + "/" + "prints_behavioural_matrix.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')

def print_network_social_component_matrix(
    fileName: str, Data, cmap_behaviour: Union[LinearSegmentedColormap,str], nrows:int, ncols:int, frames_list:list[int], round_dec:int, dpi_save:int
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

    plotName = fileName + "/Prints"
    f = plotName + "/" + "prints_network_social_component_matrix.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')

def print_network_information_provision(
    fileName: str, Data, cmap_behaviour: Union[LinearSegmentedColormap,str], nrows:int, ncols:int, frames_list:list[int], round_dec:int, dpi_save:int
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

    plotName = fileName + "/Prints"
    f = plotName + "/" + "prints_behaviour_information_provision_matrix.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')


def prints_culture_network(
    fileName: str, Data,layout:str, cmap_culture: LinearSegmentedColormap,node_size:int, nrows:int, ncols:int, norm_zero_one: SymLogNorm, frames_list:list[int], round_dec:int, dpi_save:int
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
    cbar.set_label(r"Identity, $I_{t,n}$")

    f = fileName + "/Prints/prints_culture_network.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')

def plot_average_culture_timeseries(fileName: str, Data, dpi_save:int):
    y_title = "Average Culture"
    property = "network_average_culture"

    fig, ax = plt.subplots(figsize=(10,6))
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

    plotName = fileName + "/Plots"
    f = plotName + "/" + property + "_timeseries.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')


def plot_culture_timeseries(fileName: str, Data, dpi_save:int):

    ##plot cultural evolution of agents
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Identity, $I_{t,n}$")

    data = np.asarray(Data["individual_culture"])  # bodge

    for i in range(int(int(Data["N"]))):
        # print(Data["individual_culture"][i])
        ax.plot(Data["network_time"], data[i])
    ax.axvline(Data["culture_momentum_real"], color='r',linestyle = "--")

    plotName = fileName + "/Plots"
    f = plotName + "/" + "cultural_evolution.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')

# make animate_network_social_component_matrix
def animate_network_social_component_matrix(fileName: str, Data, interval:int, fps:int, round_dec:int, cmap: Union[LinearSegmentedColormap,str], norm_zero_one):
    
    def update(i):
        M = Data["network_social_component_matrix"][i]
        # print("next frame!",M)        
        matrice.set_array(M)
        # Set the title        
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][i], round_dec))
            )
        return matrice

    fig, ax = plt.subplots(figsize=(10,6))

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
    animateName = fileName + "/Animations"
    f = animateName + "/" + "network_social_component_matrix_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)

    return ani

def animate_network_information_provision(fileName: str, Data, interval:int, fps:int, round_dec:int, cmap: Union[LinearSegmentedColormap,str]):

    def update(i):
        M = Data["behaviour_information_provision"][i]
        #print("next frame!",M, np.shape(M))  
      
        matrice.set_array(M)
        # Set the title        
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][i], round_dec))
            )
        return matrice

    fig, ax = plt.subplots(figsize=(10,6))

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
    animateName = fileName + "/Animations"
    f = animateName + "/" + "behaviour_information_provision_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)
    return ani

def prints_behaviour_timeseries_plot(fileName: str, Data, property:str, y_title:str, nrows:int, ncols:int, dpi_save:int):
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
    
    plotName = fileName + "/Plots"
    f = plotName + "/" + property + "_prints_timeseries.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')

def prints_behaviour_timeseries_plot_colour_culture(
    fileName: str, Data, property:str, y_title:str, nrows:int, ncols:int, dpi_save:int, culture_cmap, norm_zero_one
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
    cbar_culture.set_label(rr"Identity, $I_{t,n}$")

    plotName = fileName + "/Plots"
    f = plotName + "/" + property + "_prints_behaviour_timeseries_plot_colour_culture.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')


def standard_behaviour_timeseries_plot(fileName: str, Data, property:str, y_title:str, dpi_save:int):
    PropertyData = Data[property].transpose()

    fig, ax = plt.subplots(figsize=(10,6))
    for i in range(int(Data["N"])):
        for v in range(int(Data["M"])):
            ax.plot(Data["network_time"], PropertyData[i][v])
    ax.axvline(Data["culture_momentum_real"], color='r',linestyle = "--")
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    plotName = fileName + "/Plots"
    f = plotName + "/" + property + "_timeseries.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')


def plot_value_timeseries(fileName: str, Data, nrows:int, ncols:int, dpi_save:int,):
    prints_behaviour_timeseries_plot(
        fileName, Data, "behaviour_value", "Trait Value", nrows, ncols, dpi_save, 
    )


def plot_threshold_timeseries(fileName: str, Data, nrows:int, ncols:int, dpi_save:int,):
    prints_behaviour_timeseries_plot(
        fileName, Data, "behaviour_threshold", "Threshold", nrows, ncols, dpi_save
    )


def plot_attitude_timeseries(fileName: str, Data, nrows:int, ncols:int, dpi_save:int):
    
    #print(Data["behaviour_attitude"],np.shape(Data["behaviour_attitude"]))


    prints_behaviour_timeseries_plot(
        fileName, Data, "behaviour_attitude", "Attitude", nrows, ncols, dpi_save,
    )


def plot_av_carbon_emissions_timeseries(fileName: str, Data, dpi_save:int):
    y_title = "Carbon Emissions Per Individual"
    property = "individual_carbon_emissions"

    fig, ax = plt.subplots(figsize=(10,6))
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

    plotName = fileName + "/Plots"
    f = plotName + "/" + property + "_timeseries.eps"
    fig.savefig(f, dpi=dpi_save,format='eps')


"""
