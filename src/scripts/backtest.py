"""Master script for generating backtesting data."""


import argparse
import json
import os
import pandas as pd
import progressbar
import sys

sys.path.insert(0, os.path.abspath("."))

from src import load  # noqa: E402
from src import munge  # noqa: E402
import src.models.gp  # noqa: E402


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="Generate backtested predictions for each match.")
    parser.add_argument("matches_json", help="A JSON of matches.")
    parser.add_argument("method", help="Fitting method. Currently supported: "
                                       "{'newton'}")
    parser.add_argument("output", help="Output file for predictions.")
    help = "Number of initial matches to use as initial training data."
    parser.add_argument("training_matches", help=help, type=int)
    help = "Number of matches after the training matches to use for prediction."
    parser.add_argument("test_matches", help=help, type=int)
    parser.add_argument("--scale", type=float, default=2,
                        help="Scaling factor for covariance function (in "
                             "years). Default: 2 years.")
    parser.add_argument("--logistic_scale", type=float, default=5.0,
                        help="Scaling factor for the logistic win probability "
                             "function.")
    parser.add_argument("--radi_prior_sd", type=float, default=10.0,
                        help="Standard deviation of the Radiant advantage "
                             "prior Gaussian distribution.")
    args = parser.parse_args()
    args.scale *= 365 * 24 * 60 * 60 * 1000
    return args


def load_matches(json_file):
    """Load the JSON file into a matches data frame."""
    with open(json_file) as fh:
        matches = load.matches_json_to_df(json.load(fh)['data'])
    return matches


def iterative_newton_fitter(matches, args):
    """Iterative Newton-CG fitter."""
    # Fit the initial model.
    initial_matches = matches.iloc[:args.training_matches]
    initial_matches_players_mat = munge.make_match_players_matrix(
        initial_matches.radiant_players, initial_matches.dire_players)
    gp_model = src.models.gp.SkillsGPMAP(
        None,
        None,
        initial_matches_players_mat,
        initial_matches.startTimestamp.values,
        initial_matches.radiantVictory,
        "exponential",
        {"scale": args.scale},
        args.radi_prior_sd,
        args.logistic_scale)
    sys.stderr.write(f"Fitting the {args.training_matches} matches.\n")
    gp_model.fit()

    # Iteratively predict and refit the model.
    predictions = []
    for k in progressbar.progressbar(range(args.test_matches)):
        k = args.training_matches + k
        new_match = matches.iloc[[k]]

        # Add prediction of this new match.
        try:
            predictions.append(gp_model.predict_matches(
                new_match.radiant_players, new_match.dire_players,
                new_match.startTimestamp))
        except Exception as e:
            sys.stderr.write(
                f"Encountered error while predicting at iteration {k}: \n")
            sys.stderr.write(new_match.to_string())
            raise e

        # Refit with this new match.
        try:
            new_players_mat = munge.make_match_players_matrix(
                new_match.radiant_players, new_match.dire_players)
            gp_model = gp_model.add_matches(new_players_mat,
                                            new_match.startTimestamp,
                                            new_match.radiantVictory)
            gp_model.fit()
        except Exception as e:
            sys.stderr.write(
                f"Encountered error while refitting at iteration {k}: \n")
            sys.stderr.write(new_match.to_string())
            raise e
    return pd.concat(predictions)


def main():
    args = parse_args()
    matches = load_matches(args.matches_json)
    if args.method == 'newton':
        predictions = iterative_newton_fitter(matches, args)
        predictions.to_csv(args.output, sep="\t")


if __name__ == "__main__":
    main()
