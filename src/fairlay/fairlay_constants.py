"""Constants used by Fairlay."""


import pandas as pd


# Expected field names of a Fairlay "market" JSON entry.
FAIRLAY_MARKET_KEYS = set([
    'Comp',
    'Descr',
    'Title',
    'CatID',
    'ClosD',
    'SettlD',
    'Status',
    'Ru',
    '_Type',
    '_Period',
    'SettlT',
    'Comm',
    'Settler',
    'ComRecip',
    'MinVal',
    'MaxVal',
    'LastCh',
    'LastSoftCh',
    'LogBugs',
    'OrdBStr',
    'Pop',
    'Margin',
    'ID'])


MARKET_CATEGORY = pd.Series({
    1: "Soccer",
    2: "Tenis",
    3: "Golf",
    4: "Cricket",
    5: "RugbyUnion",
    6: "Boxing",
    7: "Horse Racing",
    8: "Motorsport",
    10: "Special",
    11: "Rugby League",
    12: "Bascketball",
    13: "American Football",
    14: "Baseball",
    15: "Politics",
    16: "Financial",
    17: "Greyhound",
    18: "Volleyball",
    19: "Handball",
    20: "Darts",
    21: "Bandy",
    22: "Winter Sports",
    24: "Bowls",
    25: "Pool",
    26: "Snooker",
    27: "Table tennis",
    28: "Chess",
    30: "Hockey",
    31: "Fun",
    32: "eSports",
    33: "Inplay",
    34: "reserved4",
    35: "Mixed Martial Arts",
    36: "reserved6",
    37: "reserved",
    38: "Cycling",
    39: "reserved9",
    40: "Bitcoin",
    42: "Badminton"})


MARKET_TYPE = pd.Series({
    0: "MONEYLINE",
    1: "OVER_UNDER",
    2: "OUTRIGHT",
    3: "GAMESPREAD",
    4: "SETSPREAD",
    5: "CORRECT_SCORE",
    6: "FUTURE",
    7: "BASICPREDICTION",
    8: "RESERVED2",
    9: "RESERVED3",
    10: "RESERVED4",
    11: "RESERVED5",
    12: "RESERVED6"})


MARKET_PERIOD = pd.Series({
    0: "UNDEFINED",
    1: "FT",
    2: "FIRST_SET",
    3: "SECOND_SET",
    4: "THIRD_SET",
    5: "FOURTH_SET",
    6: "FIFTH_SET",
    7: "FIRST_HALF",
    8: "SECOND_HALF",
    9: "FIRST_QUARTER",
    10: "SECOND_QUARTER",
    11: "THIRD_QUARTER",
    12: "FOURTH_QUARTER",
    13: "FIRST_PERIOD",
    14: "SECOND_PERIOD",
    15: "THIRD_PERIOD"})


# Datetime format fields.
DATETIME_COLS = set(['ClosD', 'LastCh', 'LastSoftCh', 'SettlD'])


# Base URL for the Fairlay API.
FAIRLAY_BASE_URL = 'http://31.172.83.181:8080/free1/markets/'


FAIRLAY_TEAM_NAME_TR = {
    'evos esports': 'Team EVOS',
    '2be continued esports': '2Be.Dota2',
    'prime': 'The Prime NND',
    'sunrise': 'VG.Sunrise',
    'flytomoon': 'FlyToMoon',
    'infamous': 'Infamous Gaming',
}


def fairlay_team_tr(team_name):
    """Case insensitive Fairlay to Datdota team name translation."""
    if isinstance(team_name, str):
        return FAIRLAY_TEAM_NAME_TR.get(team_name.lower(), team_name)
    elif isinstance(team_name, pd.Series):
        new_names = team_name.copy()
        notna = team_name.notna()
        new_names.loc[notna] = [FAIRLAY_TEAM_NAME_TR.get(x.lower(), x)
                                for x in team_name.loc[notna]]
        return new_names
    else:
        return [FAIRLAY_TEAM_NAME_TR.get(x, x) for x in team_name.str.lower()]
