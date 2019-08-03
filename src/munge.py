"""Mungers for data."""


import itertools
import numpy as np
import pandas as pd


def make_match_players_matrix(radiant_player_ids, dire_player_ids):
    """Return an indicator matrix for match players."""
    assert len(radiant_player_ids) == len(dire_player_ids)
    radiant_player_mat = pd.DataFrame(
        list(radiant_player_ids.apply(lambda x: {id: 1 for id in x}))).fillna(0)
    dire_player_mat = pd.DataFrame(
        list(dire_player_ids.apply(lambda x: {id: -1 for id in x}))).fillna(0)
    match_players = radiant_player_mat.add(dire_player_mat, fill_value=0.0)
    match_players.index = radiant_player_ids.index
    return match_players


def player_id_to_player_name(player_ids, player_names, team_ids, team_names):
    """Make a series of player names indexed by player IDs.

    Args:
        player_ids (list): List of lists of player IDs.
        player_names (list): List of lists of player names.
    """
    output_series = pd.DataFrame(
        {
            "name": list(itertools.chain.from_iterable(list(player_names))),
            "team": np.repeat(team_names.values, [5] * len(team_names)),
            "team_id": np.repeat(team_ids.values, [5] * len(team_ids))
        },
        index=itertools.chain.from_iterable(list(player_ids)))
    return output_series[~output_series.index.duplicated(keep='first')]


def match_df_to_player_mat(match_df):
    """Get a player matrix from the output of src.load.all_matches_df()."""
    players_mat = make_match_players_matrix(
        match_df.radiant_players, match_df.dire_players)
    return players_mat


def match_df_to_player_df(match_df):
    """Convert a match data frame into a players data frame.

    A match data frame is (for example) an output of `src.load.all_matches_df`.
    """
    players = player_id_to_player_name(
        pd.concat([match_df.radiant_players, match_df.dire_players]),
        pd.concat([match_df.radiant_nicknames, match_df.dire_nicknames]),
        pd.concat([match_df.radiant_valveId, match_df.dire_valveId]),
        pd.concat([match_df.radiant_name, match_df.dire_name]))
    return players
