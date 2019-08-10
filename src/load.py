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


def matches_json_to_df(matches_json):
    """Convert a matches JSON format into a Pandas data frame."""
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


def all_matches_df():
    """Load a Pandas DataFrame of all matches."""
    matches_json = all_matches_json()
    return matches_json_to_df(matches_json)


class MatchDF:
    """Provide error checking and functionality for match data frame."""

    def _validate_matches_df(self):
        expected_columns = [
            'startDate',
            'league_name',
            'radiant_name',
            'dire_name',
            'radiantVictory',
            'radiant_nicknames',
            'dire_nicknames',
            'dire_players',
            'dire_valveId',
            'duration',
            'league_id',
            'radiant_players',
            'radiant_valveId',
            'seriesId',
            'startTimestamp']
        if not all(np.sort(self.df.columns) == sorted(expected_columns)):
            raise ValueError("Expected columns not found.")

    def _add_series_start_time(self):
        """
        Compute and add the series start time of each match to the matches
        data frame.
        """
        start_time_of_series = pd.Series(self.df.seriesId.values,
                                         index=self.df.startTimestamp,
                                         name='series_id').dropna()
        start_time_of_series = (start_time_of_series
                                .reset_index()
                                .groupby('series_id')
                                .aggregate({'startTimestamp': min})
                                .squeeze())
        # By default, use the match start times, but fill in the series start
        # times if available.
        match_series_start_time = pd.Series(self.df.startTimestamp.values,
                                            index=self.df.seriesId,
                                            name='startTimestamp')
        idx = match_series_start_time.index.isin(start_time_of_series.index)
        match_series_start_time.loc[idx] = start_time_of_series[
            match_series_start_time.loc[idx].index]
        self.df = self.df.assign(
            series_start_time=match_series_start_time.values)

    def __init__(self, matches_df):
        self.df = matches_df.sort_index()
        self._validate_matches_df()
        self._add_series_start_time()
