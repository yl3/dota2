"""Helper script for fitting the TI9 dataset.

This script can be used for profiling.
"""


import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath("."))

from src import load
from src import munge
import src.models.gp


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="Fit and save a GP model on the TI9 matches.")
    parser.add_argument("pickle_output", help="Output file for the samples.")
    parser.add_argument("--n_iters", type=int, default=10000,
                        help="Number of iterations to run. Default: 10000.")
    parser.add_argument("--scale", type=float,
                        default=(2 * 365 * 24 * 60 * 60 * 1000),
                        help="Scaling factor for covariance function (in ms). "
                             "Default: 2 years.")
    parser.add_argument("--sampling_freq", type=int, default=100,
                        help="Save a sample every this many iterations. "
                             "Default: 100.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    matches = load.all_matches_df()
    matches_ti9 = matches.loc[matches.league_name == "The International 2019"]
    players_mat_ti9 = munge.make_match_players_matrix(
        matches_ti9.radiant_players, matches_ti9.dire_players)
    gp = src.models.gp.SkillsGP(players_mat_ti9.values,
                                matches_ti9.startTimestamp,
                                matches_ti9.radiantVictory,
                                matches_ti9.columns.values,
                                "exponential", {"scale": args.scale},
                                propose_sd=0.1,
                                save_every_n_iter=args.sampling_freq)
    gp.iterate(args.n_iters)
    with open(args.pickle_output, 'wb') as fh:
        pickle.dump(gp.samples, fh)


if __name__ == "__main__":
    main()
