"""Helper functions for analysing Fairlay odds results."""

import json
import re
import requests

import numpy as np
import scipy

from .fairlay_constants import FAIRLAY_BASE_URL


def _emp_poibin_ci(win_probs, win_amt, loss_amt, alpha):
    """
    Helper function for computing empirical PoiBin confidence intervals.
    """
    outcome = np.where(
        np.isnan(win_probs),
        np.nan,
        scipy.stats.bernoulli.rvs(np.nan_to_num(win_probs, nan=0)))
    total_reward = np.sum(np.where(outcome, win_amt, loss_amt), 1)
    return np.quantile(total_reward, (alpha, 1 - alpha))


def compute_fairlay_outcomes(fairlay_df, bool_vec):
    fairlay_df = fairlay_df.loc[bool_vec]

    # Only keep the latest odds we can find for each map.
    grp_cols = ['map_id', 'wager_type', 'RunnerName']
    fairlay_df = \
        (fairlay_df.sort_values('LastSoftCh')
         .drop(['wager_type', 'RunnerName'], 1)
         .reset_index()
         .groupby(grp_cols, as_index=False)
         .apply(lambda df: df.iloc[-1]))

    # Compute empirical confidence interval for total reward.
    N = 100000
    alpha = 0.025
    win_probs = np.repeat(
        np.where(fairlay_df.wager_type == 'on', fairlay_df.pred_win_prob,
                 1 - fairlay_df.pred_win_prob).reshape(1, -1),
        N,
        axis=0)
    empirical_ci = _emp_poibin_ci(
        win_probs,
        np.repeat(
            (fairlay_df.odds_c - 1).values.reshape(1, -1), N,
            axis=0),
        -np.ones((N, fairlay_df.shape[0])),
        alpha)

    # Compute approximated confidence interval for total reward.
    total_ev = fairlay_df.ev.sum()
    total_outcome = fairlay_df.outcome.sum()
    n_matches = fairlay_df.shape[0]
    return total_outcome, total_ev, empirical_ci, n_matches


def fairlay_outcome_printer(fairlay_df, bool_vec, title=None):
    res = compute_fairlay_outcomes(fairlay_df, bool_vec)
    total_outcome, total_ev, empirical_ci, n_matches = res

    if title is not None:
        print(title + "\n" + "-" * len(title))
    print("Maps included: {}".format(n_matches))
    print("Total EV: {:.2f} ({:.2f}, {:.2f})".format(
        total_ev,
        empirical_ci[0],
        empirical_ci[1]))
    print("Total outcome: {}".format(total_outcome))
    print("")


def fetch_markets(base_url=FAIRLAY_BASE_URL, qry_params=None):
    """Fetch 2 data from fairlay.com.

    Currently only supporting category 32, i.e. esports. See
    https://fairlay.com/api/#/data/?id=market-categories. Also, only Dota 2
    markets, i.e. those with 'dota' in the competition name, are returned.
    """
    MAX_RECORDS = 100000
    params = {'Cat': 32, 'ToID': MAX_RECORDS, 'NoZombie': True}
    if qry_params is not None:
        params.update(qry_params)
    if not base_url.endswith('/'):
        base_url += '/'
    url = base_url + json.dumps(params)
    res = requests.get(url)
    if res.status_code == 200:
        n_records = len(res.json())
        if n_records == MAX_RECORDS:
            raise ValueError('Did not fetch all records.')
        dota2_comps = [j for j in res.json()
                       if re.search("dota", j['Comp'], re.IGNORECASE)]
        return dota2_comps
    else:
        raise Exception(f"Status code is not 200: {res.text}")
