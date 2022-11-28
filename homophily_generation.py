from resources.network import Network
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from resources.plot import multi_line_matrix_plot,homophily_matrix,homophily_contour
from resources.utility import createFolder


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


params = {
    "total_time": 3000,
    "delta_t": 1.0,
    "compression_factor": 10,
    "save_data": 1, 
    "alpha_change" : 1.0,
    "degroot_aggregation": 1,
    "averaging_method": "Arithmetic",
    "phi_lower": 0.01,
    "phi_upper": 0.05,
    "N": 200,
    "M": 3,
    "K": 20,
    "prob_rewire": 0.1,
    "set_seed": 1,
    "seed_list": [1,2,3,4,5],
    "culture_momentum_real": 100,
    "learning_error_scale": 0.02,
    "discount_factor": 0.95,
    "homophily": 0.95,
    "homophilly_rate" : 1,
    "confirmation_bias": 20,
    "a_attitude": 0.5,
    "b_attitude": 0.5,
    "a_threshold": 1,
    "b_threshold": 1,
    "action_observation_I": 0.0,
    "action_observation_S": 0.0,
    "green_N": 0,
    "network_structure": "small_world"
}



if __name__ == "__main__":

    f = "results/homophily_generation"
    createFolder(f)#create folder but not saving data
    dpi_save = 1200

    homophily_list = np.linspace(0, 1, 50)
    homophily_list_label = [0.0,0.2,0.4,0.6,0.8, 1.0]
    homophily_list_pos = [0,10,20,30,40, 50]
    #homophily_list_pos_line = [0,0.2,0.4,0.6,0.8, 1.0]

    prob_rewire_list = np.linspace(0, 1, 50)
    prob_rewire_list_label =  [0.0,0.2,0.4,0.6,0.8, 1.0]
    prob_rewire_list_pos = [0,10,20,30,40, 50]
    #prob_rewire_list_pos_line = [0,0.2,0.4,0.6,0.8, 1.0]

    total_difference_init_matrix = []

    for i in homophily_list:
        params["homophily"] = i
        total_difference_init_row = []
        for j in prob_rewire_list:
            params["prob_rewire"] = j
            social_network = Network(params)
            total_difference_init_row.append(sum(social_network.total_identity_differences))
        total_difference_init_matrix.append(total_difference_init_row)

    col_label = r"Probability of re-wiring, $p_r$"
    row_label = r"Attribute homophily, $h$"
    y_label =r"Identity difference, $\Delta_I$"
    Y_param = "total_difference"
    cmap = get_cmap("Blues")

    #multi_line_matrix_plot(f,total_difference_init_matrix, prob_rewire_list, homophily_list,  Y_param, cmap, dpi_save, prob_rewire_list_pos_line, prob_rewire_list_label, homophily_list_pos_line, homophily_list_label,1, col_label, row_label, y_label)
    #homophily_matrix(f,total_difference_init_matrix,prob_rewire_list_pos,prob_rewire_list_label,homophily_list_pos,homophily_list_label,cmap, dpi_save, col_label, row_label, y_label)
    homophily_contour(f,total_difference_init_matrix,prob_rewire_list_pos,prob_rewire_list_label,homophily_list_pos,homophily_list_label,cmap, dpi_save, col_label, row_label, y_label,homophily_list,prob_rewire_list)

    plt.show()






