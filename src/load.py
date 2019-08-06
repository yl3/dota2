"""Functions for loading data etc."""


import json
import numpy as np
import pandas as pd


def all_matches_json():
    """Load all matches."""
    input_file = "data/raw/all_matches.json"
    with open(input_file) as fh:
        all_matches = json.load(fh)
    return all_matches['data']


def _flatten_match_json(json_entry):
    """Flatten a JSON entry for a match."""
    # First copy over the basic types.
    simple_keys = ["matchId", "seriesId", "startDate", "duration",
                   "radiantVictory"]
    out_dict = {key: json_entry[key] for key in simple_keys}

    # Then copy each complex attribute individually.
    for side in ("radiant", "dire"):
        out_dict[side + '_valveId'] = json_entry[side]['valveId']
        out_dict[side + '_name'] = json_entry[side]['name']
        out_dict[side + '_players'] = \
            [x['steamId'] for x in json_entry[side + "Players"]]
        out_dict[side + '_nicknames'] = \
            [x['nickname'] for x in json_entry[side + "Players"]]
    out_dict['league_id'] = json_entry['league']['leagueId']
    out_dict['league_name'] = json_entry['league']['name']
    return out_dict


def all_matches_df():
    """Load a Pandas DataFrame of all matches."""
    matches_json = all_matches_json()
    matches_df = pd.DataFrame([_flatten_match_json(x) for x in matches_json])
    matches_df['seriesId'] = matches_df['seriesId'].astype(pd.Int64Dtype())
    matches_df['startTimestamp'] = matches_df['startDate'].values
    matches_df['startDate'] = pd.to_datetime(matches_df.startDate, unit='ms')
    matches_df.set_index("matchId", inplace=True)
    col_names = ['startDate', 'league_name', 'radiant_name', 'dire_name',
                 'radiantVictory', 'radiant_nicknames', 'dire_nicknames']
    cols = np.concatenate(
        [col_names, matches_df.columns[~matches_df.columns.isin(col_names)]])
    matches_df = matches_df.loc[:, cols]
    matches_df.sort_values('startDate', inplace=True)
    return matches_df
