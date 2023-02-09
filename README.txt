Version 0.5 of the model.

[NEED TO UPDATE]

The aim of this model is to simulate how behaviours can lead to idenity change which in turn changes who indvidiuals listen to whilst learning socially. 

Outline of model:
The python files that model is built of may be found in src. network.py is the main manager of the simulation and holds a list of Individual objects (individual.py) that represent people which interact within a small world social network. Each N individuals has M behaviours which evolve over time due to the social interactions either through a DeGroot model interaction or by selecting a single indivdual who influences them that turn. The average over over M  attitudes behaviours produces an identity representing how green an indivdiual sees themselves. The distance between individuals identity then determines how strong their connection is and thus how much attentioned is paid to that neighbours opinion when learning socially.

How to run it:
To execute a single run of the simulation first edit the base_params.json file in the constants folder then go to runplot.py, here choose which plots you would like to produce and make sure RUN is set to 1 or True.

To execute multiple runs varying one parameter have a look at multi_run_single_param.py 

To execute multiple runs varying two parameters in a dependant or grid like fashion have a look at multi_run_2D_param.py

To exectute multiple runs varying n parameters independantly have a look at multi_run_n_param.py

To perform sensitivity analysis on the model see SA_sobol, this takes ages!

Other information:
 - figures are produced in plot.py which is a bit of a mess, I have left in functions that are not working at the moment (commented out) for the sake of inspiration and the plots are stored in either png or eps
 - its not a particularly rapid model to run for lots of time steps
 - To use the load (or not RUN) functionality paste in the location of the folder name of the date you want to use. This is printed when you first run that set up so if say you want to test out different graph aesthetic you can quickly test this without re-running the simulation.
 
For any questions please contact Daniel Torren Peraire at Daniel.Torren@uab.cat or dtorrenp@hotmail.com




