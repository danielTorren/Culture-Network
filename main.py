import argparse
#generating_data
from generating_data.alpha_change_micro_consensus_gen import main as alpha_change_micro_consensus_gen
from generating_data.bifurcation_gen import main as bifurcation_gen
from generating_data.oneD_param_sweep_gen import main as oneD_param_sweep_gen
from generating_data.sensitivity_analysis_gen import main as sensitivity_analysis_gen
from generating_data.single_shot_gen import main as single_shot_gen
from generating_data.splitting_eco_warrior_gen import main as splitting_eco_warrior_gen
from generating_data.twoD_param_sweep_gen import main as twoD_param_sweep_gen
#plotting_data
from plotting_data.alpha_change_micro_consensus_plot import main as alpha_change_micro_consensus_plot
from plotting_data.bifurcation_plot import main as bifurcation_plot
from plotting_data.oneD_param_sweep_plot import main as oneD_param_sweep_plot
from plotting_data.sensitivity_analysis_plot import main as sensitivity_analysis_plot
from plotting_data.single_shot_plot import main as single_shot_plot
from plotting_data.splitting_eco_warrior_plot import main as splitting_eco_warrior_plot
from plotting_data.twoD_param_sweep_plot import main as twoD_param_sweep_plot
from plotting_data.hyperbolic_discounting_example import main as hyperbolic_discounting_example
from plotting_data.init_distribution_example import main as init_distribution_example

parser=argparse.ArgumentParser()

parser.add_argument("--experiment_type", "-rt", default='single_shot', type=str)
parser.add_argument("--RUN_NAME", default='SINGLE', type=str)
parser.add_argument("--RUN_TYPE", default=0, type=int)
parser.add_argument("--nrows_gen", default=2, type=int)
parser.add_argument("--ncols_gen", default=3, type=int)
parser.add_argument("--N_samples ", default=1024, type=int)
parser.add_argument("--calc_second_order", default=False, type=bool)
parser.add_argument("--cluster_count_run", default=False, type=bool)
parser.add_argument("--culture_run", default=False, type=bool)


parser.add_argument("--", default=, type=)
parser.add_argument("--", default=, type=)
parser.add_argument("--", default=, type=)
parser.add_argument("--", default=, type=)
parser.add_argument("--", default=, type=)
parser.add_argument("--", default=, type=)


args = parser.parse_args()

if args.run_type == 'single_shot':
    single_shot_gen(run=args.run, load=args.load)
elif args.run_type == 'sensitivity_analysis':
    sensitivity_analysis(run=args.run, load=args.load)

