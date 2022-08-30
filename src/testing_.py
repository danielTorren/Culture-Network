import matplotlib.pyplot as plt
import numpy as np
from typing import Union
import pickle
from plot import plot_alpha_variation
from utility import createFolderSA

##inital distribution parameters - doing the inverse inverts it!
alpha_attract = 3
beta_attract = 2

alpha_threshold = 3
beta_threshold = 2

bin_num = 1000
num_counts = 1000


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



if __name__ == '__main__':

    "Plotting inital distributions"
    #plot_beta(
    #    alpha_attract, beta_attract, alpha_threshold, beta_threshold, bin_num, num_counts
    #)
    

    "Test loading data"
    fileName = "results/_DEGROOT_10000_3_100_0.05_10_0.1_1_0.02_1_1_1_1_10"

    #file = open(fileName + "/Data.pkl",'rb')
    #object_file = pickle.load(file)
    #file.close()
    #print(object_file.agent_list)
    #print(object_file.history_average_culture)

    "Running matrix based simulation"

    "alpha variation"
    phi_list = [-1,0,1]
    dpi_save = 1200
    plot_alpha_variation(fileName,num_counts,phi_list,dpi_save)
    plt.show()

