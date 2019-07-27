"""Helper script for fitting the TI9 dataset.

This script can be used for profiling.
"""


import argparse
import os
import dill
import sys

sys.path.insert(0, os.path.abspath("."))

from src import load  # noqa: E402
from src import munge  # noqa: E402
import src.models.gp  # noqa: E402


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="Fit and save a GP model on matches.")
    parser.add_argument("dill_output", help="Output file for the fitted model.")
    parser.add_argument("--subset", default=None,
                        help="Optional match subset to use. Currently "
                             "supported: ti9.")
    parser.add_argument("--n_iters", type=int, default=10000,
                        help="Number of iterations to run. Default: 10000.")
    parser.add_argument("--scale", type=float, default=2,
                        help="Scaling factor for covariance function (in "
                             "years). Default: 2 years.")
    parser.add_argument("--sampling_freq", type=int, default=100,
                        help="Save a sample every this many iterations."
                             "Default: 100.")
    parser.add_argument("--iter_method", default="full",
                        help="Iteration method for gp.iterate()")
    parser.add_argument("--logistic_scale", type=float, default=5.0,
                        help="Scaling factor for the logistic win probability "
                             "function.")
    args = parser.parse_args()
    args.scale *= 365 * 24 * 60 * 60 * 1000
    return args


def main():
    args = parse_args()
    matches = load.all_matches_df()
    if args.subset is not None:
        if args.subset == "ti9":
            matches = \
                matches.loc[matches.league_name == "The International 2019"]
        else:
            raise ValueError(f"--subset '{args.subset}' is not recognised.")
    players_mat = munge.make_match_players_matrix(
        matches.radiant_players, matches.dire_players)
    gp = src.models.gp.SkillsGP(players_mat.values,
                                matches.startTimestamp.values,
                                matches.radiantVictory,
                                matches.columns.values,
                                "exponential", {"scale": args.scale},
                                propose_sd=0.1,
                                save_every_n_iter=args.sampling_freq)
    gp.iterate(args.n_iters, method=args.iter_method)
    with open(args.dill_output, 'wb') as fh:
        dill.dump(gp, fh)


if __name__ == "__main__":
    main()
