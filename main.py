import argparse
from single_shot import main as single_shot
from sensitivity_analysis import main as sensitivity_analysis
parser=argparse.ArgumentParser()




parser.add_argument("--run_type", "-rt", default='single_shot', type=str)
parser.add_argument("--run", "-r", default=True, type=bool)
parser.add_argument("--load", "-l", defa)

args = parser.parse_args()

if args.run_type == 'single_shot':
    single_shot(run=args.run, load=args.load)
elif args.run_type == 'sensitivity_analysis':
    sensitivity_analysis(run=args.run, load=args.load)

