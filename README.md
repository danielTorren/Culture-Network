## Folder structure:
Inside the package folder, you will find a set of folders that include the core model itself, running, plotting and other utility code. Additionally, there are two jupyter notebooks. The first of which, "produce_figures.ipynb" is a guide to reproduce the figures found in the paper, some of these require substantial run times. Secondly, there is "model_playground.ipynb" which allows you to test out a single model run for a variety of different parameter inputs and produce different plots and animations to analyse that experiment.

## Outline of model:
The python files that the core model is built of may be found in package/model, network.py is the main manager of the simulation and holds a list of Individual objects (individual.py) that represent people which interact within a small world social network. Each of the N individuals has M behaviours which evolve due to imperfect social interactions. The time-discounted average-over-M attitudes produce an identity representing how green individuals see themselves. The distance between individuals' environmental identities then determines how strong their connection is and thus how much attention is paid to that neighbour's opinion.

## Other folders in the package:
- "package/constants" contains several json files. "base_params.json" contains the default model parameters which are used to reproduce multiple figures. Variable parameter json files which are used to set the ranges of parameter variations for the sensitivity analysis (variable_parameters_dict_SA.json) or which two parameters to vary to cover a 2D parameter space (variable_parameters_dict_2D.json).

- "generating_data" contains several python files that load in inputs and run the model for said conditions, then save this data. "single_experiment_gen.py" runs a single experiment, "oneD_param_sweep_gen.py" runs multiple experiments whilst varying a single parameter, "bifurcation_gen.py" runs experiments for conditions with and without behavioural interdependency, "sensitivity_analysis_gen.py" runs the model for a large number of parameter values and over multiple stochastic initial conditions, "identity_frequency_gen.py" runs three experiments with each different with different identity updating frequency, "adding_green_influencers_gen.py" runs the default model but adds green influencers and "twoD_param_sweep_gen.py" runs experiments varying two parameters to cover a two-dimensional parameter space.

- "plotting_data" loads the model results created in the "generating_data" folder, analyzes them and calls the plot functions.

- "resources" contains code that is used frequently such as saving or loading data (utility.py), running the simulation for a specific number of time steps (run.py) and plots that may be used by several files in "plotting_data", (plot.py).


## Set up steps for code and jupyter notebooks:
- (Note that the model was built with python3.9)
1. Clone the repository.
2. Create a virtual environment inside the repository and activate it (may need to pip install virtualenv) 
4. Install the necessary python libraries using pip:
	```
	python -m pip install -r requirements.txt
	```
5. Attach the jupyter notebook kernel to the virtual environment
	```
	python -m ipykernel install --name=name_of_your_virtual_environment
	```
6. Now open the jupyter notebooks
	```
	python -m jupyter notebook
	```  
7. Finally, have a play around with both the model parameters in "model_playground.ipynb" and reproduce paper figures using "produce_figures.ipynb".