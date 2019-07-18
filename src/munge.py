"""Mungers for data."""


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
