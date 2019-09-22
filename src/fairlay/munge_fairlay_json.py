"""Mungers for scraped Myfairlay data."""

import json
import numpy as np
import pandas as pd

from . import fairlay_constants
from .fairlay_constants import (  # noqa
    FAIRLAY_MARKET_KEYS,
    DATETIME_COLS,
    MARKET_PERIOD,
    MARKET_TYPE,
    MARKET_CATEGORY)


def validate_fairlay_json_elem(json_elem):
    """Verify that input is json (dictionary) with the right keys."""
    if not isinstance(json_elem, dict):
        raise TypeError("json_elem must be a dictionary.")
    if not FAIRLAY_MARKET_KEYS == set(json_elem.keys()):
        raise ValueError(f"Key mismatch. Expected keys: {FAIRLAY_MARKET_KEYS}.")


def explode_ordbstr(ordbstr):
    """Explode the OrdBStr field of a Fairlay markets JSON element.

    The string, ordbstr, is expected to be a '~'-delimited string of
    dictionaries (represented as a string.)
    """
    if not isinstance(ordbstr, str):
        raise TypeError("ordbstr must be a string.")
    ordb_substrs = ordbstr.split("~")
    order_list = [json.loads(o) if o != '' else None for o in ordb_substrs]
    return order_list


def ask_to_odds(ask_price):
    """Convert ask price to decimal odds."""
    odds = ask_price / (ask_price - 1)
    return odds


def odds_to_prob(odds):
    """Convert decimal odds to the breakeven win probability."""
    return 1/odds


def fairlay_dict_list_to_df(dict_list):
    """Convert a list of "market" dictionaries to a Pandas data frame."""
    df = pd.DataFrame(dict_list)
    df.set_index("ID", inplace=True)

    # Convert datetime columns to datetime type.
    for col in DATETIME_COLS:
        df[col] = (pd.to_datetime(df[col], utc=True)
                   .dt.round('1s')
                   .dt.tz_convert('US/Eastern'))

    # Decode some fields.
    df['MarketType'] = MARKET_TYPE[df['_Type']].values
    df['MarketPeriod'] = MARKET_PERIOD[df['_Period']].values
    df['MarketCat'] = MARKET_CATEGORY[df['CatID']].values

    # Move columns of interest to the front of the data frame.
    return df


def _dota_market_type(title, descr):
    """
    Determine the Fairlay Dota 2 market type based on title and description.
    """
    dota_market_type = \
        np.where(title.str.contains(r"kills", case=False), "kills",
        np.where(descr.str.contains("Game duration"), "duration",
        np.where(descr.str.startswith("Match Spread for HomeTeam"),
                                      "series_spread",
        np.where(descr.str.endswith(" Map"), "map",
        np.where(descr.str.contains("Winner.+-way", case=False), "series",
        np.where(descr == "Futures", "futures", "other"
    ))))))  # noqa
    return dota_market_type


def _reorder_fairlay_df_cols(df):
    """Reorder the columns in a Fairlay marker data frame."""
    COLS_OF_INTEREST = ["Comp", "Title", "Descr", "dota_market_type",
                        "LastSoftCh", "ClosD",  "wager_type", "RunnerName",
                        "handicap", "odds", 'odds_c', "amount", "winp",
                        "winp_c", "RunnerVolMatched"]
    if not set(COLS_OF_INTEREST) <= set(df.columns):
        raise ValueError("Some of the expected columns are not in df.""")
    other_cols = list(df.columns[~df.columns.isin(COLS_OF_INTEREST)])
    df = df.loc[:, COLS_OF_INTEREST + other_cols]
    return df


def compute_normalised_team_names(df):
    """Normalise team 1, team 2 and runner names in a Fairlay data frame."""
    teams = df.Title.str.extract(r'^(.+) vs. (.+)$')
    teams.columns = ['team_1', 'team_2']
    for team in teams.columns:
        teams.loc[:, team] = fairlay_constants.fairlay_team_tr(
            teams.loc[:, team])
    teams['norm_runner'] = fairlay_constants.fairlay_team_tr(df['RunnerName'])
    return teams


def compute_map_i(df):
    """Compute map index from a Fairlay data frame."""
    map_i = np.where(
        df['Descr'].str.match(r'^(1st|2nd|3rd|4th|5th) Map'),
        df['Descr'].str[0].astype(int) - 1,
        np.nan
    ).astype(pd.Int64Dtype)
    return map_i


def fairlay_json_to_df(fairlay_json):
    """Convert a Fairlay JSON markets dump into a Pandas data frame."""
    EXPLODABLE_COLS = ['Ru', 'OrdBStr']

    dict_rows = []  # Row of dictionaries to be passed to pd.DataFrame().
    for elem in fairlay_json:
        validate_fairlay_json_elem(elem)

        # Save the constant fields.
        other_cols = elem.copy()
        for key in EXPLODABLE_COLS:
            other_cols.pop(key, None)

        # Process the fields to be exploded.
        for ordb, runner in zip(explode_ordbstr(elem['OrdBStr']), elem['Ru']):
            if ordb is None:
                continue
            # for bid, ask in zip(ordb['Bids'], ordb['Asks']):
            for bid in ordb['Bids']:
                new_row = other_cols.copy()
                new_row.update({"Runner" + k: v for k, v in runner.items()})
                new_row['wager_type'] = "on"
                new_row['odds'] = bid[0]
                new_row['amount'] = bid[1]
                new_row['OrdBStr_S'] = ordb['S']
                dict_rows.append(new_row)
            for ask in ordb['Asks']:
                new_row = other_cols.copy()
                new_row.update({"Runner" + k: v for k, v in runner.items()})
                new_row['wager_type'] = "against"
                new_row['odds'] = ask_to_odds(ask[0])
                new_row['amount'] = ask[1]
                new_row['OrdBStr_S'] = ordb['S']
                dict_rows.append(new_row)

    df = fairlay_dict_list_to_df(dict_rows)

    # Compute some additional columns.
    df['dota_market_type'] = _dota_market_type(df.Title, df.Descr)
    df['RunnerName'] = df['RunnerName'].str.strip()
    pat = r" ([+-]?\d+\.5)$"
    df['handicap'] = df.RunnerName.str.extract(pat).astype(np.float).fillna(0.0)
    df['RunnerName'] = df['RunnerName'].str.replace(pat, "")
    df['map_i'] = compute_map_i(df)

    # Breakeven win probabilities with and without the commission.
    df['winp'] = (1 / df['odds'])
    df['winp'] = df['winp'].where(df['wager_type'] == "on", 1 - df['winp'])
    odds_c = (1 + (df['odds'] - 1) * (1 - df['Comm']))
    df['odds_c'] = odds_c
    df['winp_c'] = 1 / odds_c
    df['winp_c'] = df['winp_c'].where(df['wager_type'] == "on",
                                      1 - df['winp_c'])
    # Perform some rounding.
    df['odds'] = df['odds'].round(3)
    df['odds_c'] = df['odds_c'].round(3)
    df['winp'] = df['winp'].round(4)
    df['winp_c'] = df['winp_c'].round(4)

    # Convert a few dictionary columns into strings.
    for colname in ['ComRecip', 'Settler']:
        df[colname] = df[colname].astype(str)

    norm_teams = compute_normalised_team_names(df)
    df = pd.concat([df, norm_teams])
    df = _reorder_fairlay_df_cols(df)
    return df
