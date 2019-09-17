"""Static variables."""

import pandas as pd


TI9_TEAM_NAMES = {
    "Alliance": 111474,
    "Chaos Esports Club": 7203342,
    "Evil Geniuses": 39,
    "Fnatic": 350190,
    "Newbee": 6214538,
    "Infamous Gaming": 2672298,
    "KEEN GAMING": 2626685,
    "Mineski": 543897,
    "Natus Vincere": 36,
    "Ninjas in Pyjamas": 6214973,
    "OG": 2586976,
    "PSG.LGD": 15,
    "Royal Never Give Up": 6209804,
    "Team Liquid": 2163,
    "Team Secret": 1838315,
    "TNC Predator": 2108395,
    "Vici Gaming": 726228,
    "Virtus.pro": 1883502,
}


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
        return pd.Series(
            [FAIRLAY_TEAM_NAME_TR.get(x.lower(), x) for x in team_name],
            index=team_name.index)
    else:
        return [FAIRLAY_TEAM_NAME_TR.get(x, x) for x in team_name.str.lower()]
