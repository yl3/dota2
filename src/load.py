"""Functions for loading data etc."""


import json
import numpy as np
import pandas as pd
from . import munge


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

    # Strip all string columns.
    for colname in ['league_name', 'radiant_name', 'dire_name']:
        matches_df[colname] = matches_df[colname].str.strip()
    return matches_df


def all_matches_df():
    """Load a Pandas DataFrame of all matches."""
    matches_json = all_matches_json()
    return matches_json_to_df(matches_json)


class MatchDF:
    """Provide error checking and functionality for match data frame."""

    def __init__(self, matches_df):
        self._validate_matches_df(matches_df)
        self.df = matches_df
        self.df = self._preprocess_matches_df(self.df)

    @property
    def players(self):
        if not hasattr(self, '_players'):
            self._players = self._compute_players(self.df)
        return self._players

    @property
    def teams(self):
        if not hasattr(self, '_teams'):
            self._teams = munge.match_df_to_team_series(self.df)
        return self._teams

    @property
    def players_mat(self):
        if not hasattr(self, '_players_mat'):
            self._players_mat = munge.match_df_to_player_mat(self.df)
        return self._players_mat

    def loc_team(self, team_ids, and_operator=False):
        """Return a Boolean index for the matches involving given teams."""
        if and_operator:
            idx = (self.df.radiant_valveId.isin(team_ids)
                   & self.df.dire_valveId.isin(team_ids))
        else:
            idx = (self.df.radiant_valveId.isin(team_ids)
                   | self.df.dire_valveId.isin(team_ids))
        return idx

    def _validate_matches_df(self, matches_df):
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
        if not all([x in matches_df.columns for x in expected_columns]):
            raise ValueError("Expected columns not found.")

    def _fill_missing_series_id(self, matches_df):
        """Fill in missing series ID with the minus match ID."""
        new_matches_df = matches_df.copy()
        new_matches_df.loc[:, "seriesId"].fillna(-matches_df.index.to_series(),
                                                 inplace=True)
        return new_matches_df

    def _add_series_start_time(self, matches_df):
        """Compute and add the series start time of each match to self.df."""
        series_start_time = \
            matches_df.groupby('seriesId')['startTimestamp'].min()
        matches_df['series_start_time'] = \
            series_start_time[matches_df.seriesId].values
        return matches_df

    def _add_match_i_in_series(self, matches_df):
        """Add match number of each match in each series."""
        # Use the (negative) match ID as a placeholder when series ID is
        # missing.
        assert all(matches_df.index == matches_df.index.sort_values())
        temp_match_i_df = (matches_df.groupby('seriesId')
                           .apply(lambda x: pd.Series(np.arange(len(x)),
                                                      index=x.index))
                           .reset_index())
        temp_match_i_series = temp_match_i_df.set_index('matchId')[0]
        matches_df = matches_df.assign(match_i_in_series=temp_match_i_series)
        return matches_df

    def _preprocess_matches_df(self, matches_df):
        new_matches_df = matches_df.sort_index()  # Sort by match ID.
        new_matches_df = self._fill_missing_series_id(new_matches_df)
        new_matches_df = self._add_series_start_time(new_matches_df)
        new_matches_df = self._add_match_i_in_series(new_matches_df)
        return new_matches_df

    def _compute_players(self, matches_df):
        """Create a players matrix of a matches data frame."""
        return munge.match_df_to_player_df(matches_df)

    def _compute_series(self, matches_df):
        """Create a series data frame of a matches data frame."""
        # Make sure the matches are ordered by start time.
        assert all(matches_df.startTimestamp
                   == matches_df.startTimestamp.sort_values().values)
