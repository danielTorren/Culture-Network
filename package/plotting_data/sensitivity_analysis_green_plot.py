"""Performs sobol sensitivity analysis on the model. 

Created: 10/10/2022
"""

# imports
from package.plotting_data.sensitivity_analysis_plot import main

if __name__ == '__main__':
    main(
    fileName = "results\SA_AV_reps_5_samples_15360_D_vars_13_N_samples_1024",
    plot_outputs = ['emissions','var',"emissions_change"],
    dpi_save = 1200,
    latex_bool = 0,
        plot_dict = {
        "emissions_flow": {"title": r"$E_F/NM$", "colour": "red", "linestyle": "--"},
        "mu": {"title": r"$\mu_{I}$", "colour": "blue", "linestyle": "-"},
        "var": {"title": r"$\sigma^{2}_{I}$", "colour": "green", "linestyle": "*"},
        "coefficient_of_variance": {"title": r"$\sigma/\mu$","colour": "orange","linestyle": "-.",},
        "emissions_change": {"title": r"$\Delta E/NM$", "colour": "brown", "linestyle": "-*"},
        "emissions_stock": {"title": r"$E_S/NM$", "colour": "black", "linestyle": "--"},
    },
    titles = [
        r"Number of individuals, $N$", 
        r"Number of green influencers, $N_G$", 
        r"Number of behaviours, $M$", 
        r"Mean neighbours, $K$",
        r"Cultural inertia, $\rho$",
        r"Social learning error, $ \sigma_{ \varepsilon}$ ",
        r"Initial attitude Beta, $a_A$",
        r"Initial attitude Beta, $b_A$",
        r"Initial threshold Beta, $a_T$",
        r"Initial threshold Beta, $b_T$",
        r"Discount factor, $\delta$",
        r"Attribute homophily, $h$",
        r"Confirmation bias, $\theta$"
    ]
)
