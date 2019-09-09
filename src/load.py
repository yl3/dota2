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


class MatchupDict:
    """
    A helper class for querying matches using inexact time stamps. The values
    stored are pandas indices.

    Internally, the time index is stored as UTC and the team names are stored
    as lower case values.

    Args:
        team_1 (array-like): A 1D-array of team 1 identifiers or names.
        team_2 (array-like): A 1D-array of team 2 identifiers or names.
        map_i (int): Zero-based index of the current map in the series.
        time (pandas.Series): The start time of each map. The index of this
            series is what is returned when this class is queried.
    """
    # Internally, the win probabilities are stored in a dictionary of
    # (team_1, team_2, map_index) -> DataFrame, where the data frame is indexed
    # by datetime. team_1 and team_2 are normalised such that team_1 <= team_2.

    def __init__(self, team_1, team_2, map_i, time):
        self.data = {}
        team_1 = pd.Series(team_1).str.lower()
        team_2 = pd.Series(team_2).str.lower()
        norm_team_1 = np.where(team_1 <= team_2, team_1, team_2)
        norm_team_2 = np.where(team_1 <= team_2, team_2, team_1)
        time = self._utc_localise_time(time)
        temp_df = pd.DataFrame(
            {'team_1': norm_team_1, 'team_2': norm_team_2, 'map_i': map_i,
             'map_idx': time.index})
        temp_df.index = time.values
        for key, sub_df in temp_df.groupby(['team_1', 'team_2', 'map_i']):
            self.data[key] = sub_df

    def from_match_df(matchdf_obj):
        """Initialise from a MatchDF object."""
        if not isinstance(matchdf_obj, MatchDF):
            raise TypeError('matchdf_obj is not an object of MatchDF.')
        output = MatchupDict(matchdf_obj.df.radiant_name,
                             matchdf_obj.df.dire_name,
                             matchdf_obj.df.match_i_in_series,
                             matchdf_obj.df.startDate)
        return output

    def query(self, team_1, team_2, map_i, time, method='nearest',
              tolerance=None):
        """Get the match index with inexact query time.

        Args:
            team_1 (str): ID of the first team.
            team_2 (str): ID of the second team.
            map_i (int): Zero-based index of the current map in the series.
            time (numpy.datetime64): Query time for the match.
            method (str): Argument for :func:`pandas.Index.get_loc`.
            tolerance (float): Argument for :func:`pandas.Index.get_loc`.

        Returns:
            tuple: A tuple of (<map_idx>, <flipped>).
        """
        time = pd.to_datetime(time)
        team_1 = team_1.lower()
        team_2 = team_2.lower()
        if time.tz is None:
            time = time.tz_localize('UTC')
        else:
            time = time.tz_convert('UTC')
        if team_1 <= team_2:
            key = (team_1, team_2, map_i)
            flipped = False
        else:
            key = (team_2, team_1, map_i)
            flipped = True
        if key not in self.data:
            ret = (np.nan, np.nan)
        else:
            df = self.data[key]
            idx = df.index.get_loc(time, method=method, tolerance=tolerance)
            assert isinstance(idx, int)
            map_idx = df.iloc[idx]['map_idx']
            ret = (map_idx, flipped)
        return ret

    def query_l(self, team_1, team_2, map_i, time, method='nearest',
                tolerance=None):
        """Get a list of win probabilities with inexact query times.

        Args:
            team_1 (str, int or array-like): IDs of the first teams.
            team_2 (str, int or array-like): IDs of the second teams.
            map_i (int): or array-like Zero-based indexes of the maps in the
                series.
            time (numpy.datetime64): Query times for the matches.
            method (str): Argument for :func:`pandas.Index.get_loc`.
            tolerance (float): Argument for :func:`pandas.Index.get_loc`.

        Returns:
            array-like: A list of win probabilities. If `time` is a
                :class:`pandas.Series`, then return a :class:`pandas.Series`
                with the same index. Otherwise return a :class:`numpy.ndarray`.
        """
        # Use Pandas' broadcasting to fill in scalars as lists.
        # Ensure that at least one of the variables are lists to pacify Pandas.
        if isinstance(time, str):
            time = [time]
        try:
            _ = (i for i in time)
        except TypeError:
            time = [time]
        zipped_params = pd.DataFrame(
            {'qry_team1': team_1, 'qry_team2': team_2, 'qry_map_i': map_i,
             'qry_time': time})
        qry_res = [self.query(t1, t2, mi, time, method, tolerance)
                   for t1, t2, mi, time in zipped_params.itertuples(False)]
        ret_df = pd.DataFrame(
            qry_res,
            index=pd.MultiIndex.from_frame(zipped_params),
            columns=['map_id', 'flipped'])
        ret_df['map_id'] = ret_df['map_id'].astype(pd.Int64Dtype())
        return ret_df

    def _utc_localise_time(self, time):
        """Make sure time is a UTC-localised Pandas series."""
        new_time = pd.to_datetime(time)
        if new_time.dt.tz is not None:
            new_time = new_time.dt.tz_convert(None)
        return new_time


class MatchDF:
    """Provide error checking and functionality for match data frame."""

    def __init__(self, matches_df):
        self._validate_matches_df(matches_df)
        self.df = matches_df
        self.df = self._preprocess_matches_df(self.df)

    def from_json(matches_json):
        """Create a MatchDF object from a Datdota matches JSON object."""
        matches_df = matches_json_to_df(matches_json)
        return MatchDF(matches_df)

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

    def loc_team(self, *team_ids, and_operator=False):
        """Return a Boolean index for the matches involving given teams."""
        if and_operator:
            idx = (self.df.radiant_valveId.isin(team_ids)
                   & self.df.dire_valveId.isin(team_ids))
        else:
            idx = (self.df.radiant_valveId.isin(team_ids)
                   | self.df.dire_valveId.isin(team_ids))
        return idx

    def query_maps(self, team_1, team_2, map_i, time, method='nearest',
                   tolerance=None):
        """Match maps based on teams involved and inexact start time.

        See :meth:`MatchupDict.query_l` for details.

        Returns:
            pandas.DataFrame: The current data frame matched with the query.
        """
        if not hasattr(self, '_matchup_dict'):
            self._matchup_dict = MatchupDict.from_match_df(self)
        temp = self._matchup_dict.query_l(team_1, team_2, map_i, time, method,
                                          tolerance)
        out_df = self.df.loc[temp.map_id].reset_index().assign(
            qry_flipped=temp.flipped.values)
        out_df.index = temp.index
        return out_df

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
