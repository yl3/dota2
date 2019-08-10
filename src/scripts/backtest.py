"""Master script for generating backtesting data."""


import argparse
import datetime
import itertools
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
    parser.add_argument("--chunk_size", type=int, default=1,
                        help="Chunk size for iterative Newton-CG. Default: 1.")
    args = parser.parse_args()
    args.scale *= 365 * 24 * 60 * 60 * 1000
    return args


def load_matches(json_file):
    """Load the JSON file into a matches data frame."""
    with open(json_file) as fh:
        matches = load.matches_json_to_df(json.load(fh)['data'])
    return matches


def _cur_time():
    return datetime.datetime.now()


def _players_of_series(series_df):
    """Get the player IDs of a series data frame."""
    players_by_series = series_df.radiant_players + series_df.dire_players
    players_list = \
        list(itertools.chain.from_iterable(list(players_by_series)))
    return players_list


def _chunk_matches(series_list, chunk_size):
    """
    Combine series data frames as long as they do not contain the same
    players.

    Both input and outputs are lists of series data frames.
    """
    output_series_dfs = [[]]
    cur_players = []
    cur_idx = 0
    cur_chunk_size = 0
    while cur_idx < len(series_list):
        # If there is an overlap in players between series, start a new
        # output chunk.
        player_olap = (pd.Series(_players_of_series(series_list[cur_idx]))
                       .isin(cur_players).any())
        if player_olap or cur_chunk_size == chunk_size:
            output_series_dfs.append([])
            cur_players = []
            cur_chunk_size = 0
        output_series_dfs[-1].append(series_list[cur_idx])
        cur_players += _players_of_series(series_list[cur_idx])
        cur_chunk_size += 1
        cur_idx += 1
    output_series_dfs = [pd.concat(x) for x in output_series_dfs]
    return output_series_dfs


def iterative_newton_fitter(matches, args):
    """Iterative Newton-CG fitter.

    Matches are fitted in chunks based on the start time of the `series`.
    """
    # Order by series start time.
    matches = matches.sort_values('series_start_time')

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
    test_match_groups = [x[1]
                         for x in test_matches.groupby('series_start_time')]
    # Create larger chunks of series for fitting.
    test_match_groups = _chunk_matches(test_match_groups, args.chunk_size)
    predictions = []
    mu_mats = []
    var_mats = []
    matches_fitted = 0
    for match_grp in test_match_groups:
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
            mu_mat.index = predicted.index
            var_mat.index = predicted.index
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
    # Perform checks and compute series start times.
    matches = load.MatchDF(matches).df
    if args.method == 'newton':
        predictions, mu_df, var_df = iterative_newton_fitter(matches, args)
        predictions.to_csv(args.output_prefix + ".win_probs", sep="\t")
        mu_df.to_csv(args.output_prefix + ".player_skills", sep="\t")
        var_df.to_csv(args.output_prefix + ".player_skill_vars", sep="\t")


if __name__ == "__main__":
    main()
