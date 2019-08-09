"""Master script for generating backtesting data."""


import argparse
import datetime
import json
import os
import pandas as pd
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
    parser.add_argument("output_prefix", help="Output prefix for predictions.")
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


def _order_matches_by_series(matches):
    """Order a matches data frame by when the match or series started.

    A matches data frame is created using load_matches().
    """
    start_time_of_series = pd.Series(matches.seriesId.values,
                                     index=matches.startTimestamp,
                                     name='series_id').dropna()
    start_time_of_series = (start_time_of_series
                            .reset_index()
                            .groupby('series_id')
                            .aggregate({'startTimestamp': min})
                            .squeeze())
    # By default, use the match start times, but fill in the series start times
    # if available.
    match_series_start_time = pd.Series(matches.startTimestamp.values,
                                        index=matches.seriesId,
                                        name='startTimestamp')
    idx = match_series_start_time.index.isin(start_time_of_series.index)
    match_series_start_time.loc[idx] = start_time_of_series[
        match_series_start_time.loc[idx].index]
    matches = matches.assign(series_start_time=match_series_start_time.values)
    matches.sort_values('series_start_time', inplace=True)
    return matches


def _cur_time():
    return datetime.datetime.now()


def iterative_newton_fitter(matches, args):
    """Iterative Newton-CG fitter.

    Matches are fitted in chunks based on the start time of the `series`.
    """
    # Order by series start time.
    matches = _order_matches_by_series(matches)

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
    sys.stderr.write(
        f"[{_cur_time()}] Fitting the {args.training_matches} matches.\n")
    gp_model.fit()

    # Iteratively predict and refit the model.
    test_matches = matches.iloc[args.training_matches:]
    test_match_groups = test_matches.groupby('series_start_time')
    predictions = []
    mu_mats = []
    var_mats = []
    matches_fitted = 0
    for name, match_grp in test_match_groups:
        msg = (f"[{_cur_time()}] So far predicted: {matches_fitted} matches. "
               f"Predicting matches {list(match_grp.index)} "
               f"(series {list(match_grp.seriesId.unique())})...\n")
        sys.stderr.write(msg)

        # Add prediction of this new match.
        try:
            predicted, mu_mat, var_mat = gp_model.predict_matches(
                match_grp.radiant_players,
                match_grp.dire_players,
                match_grp.startTimestamp)
            predicted.reset_index(inplace=True)
            predicted.index = match_grp.index
            mu_mats.append(mu_mat)
            var_mats.append(var_mat)
            predictions.append(predicted)
        except Exception as e:
            sys.stderr.write(
                f"Encountered error after {matches_fitted} predicted: \n")
            sys.stderr.write(match_grp.to_string())
            raise e

        # Refit with this new match.
        try:
            new_players_mat = munge.make_match_players_matrix(
                match_grp.radiant_players, match_grp.dire_players)
            gp_model = gp_model.add_matches(new_players_mat,
                                            match_grp.startTimestamp,
                                            match_grp.radiantVictory)
            gp_model.fit()
        except Exception as e:
            sys.stderr.write(
                f"Encountered error after {matches_fitted} predicted: \n")
            sys.stderr.write(match_grp.to_string())
            raise e

        matches_fitted += match_grp.shape[0]
        if matches_fitted >= args.test_matches:
            break
    predictions_df = pd.concat(predictions)
    mu_df = pd.concat(mu_mats)
    var_df = pd.concat(var_mats)
    return predictions_df, mu_df, var_df


def main():
    args = parse_args()
    matches = load_matches(args.matches_json)
    if args.method == 'newton':
        predictions, mu_df, var_df = iterative_newton_fitter(matches, args)
        predictions.to_csv(args.output_prefix + ".win_probs", sep="\t")
        mu_df.to_csv(args.output_prefix + ".player_skills", sep="\t")
        var_df.to_csv(args.output_prefix + ".player_skill_vars", sep="\t")


if __name__ == "__main__":
    main()
