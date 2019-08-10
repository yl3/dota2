"""Functions for statistical analysis of the dataset."""


import numpy as np
import pandas as pd
from poibin import PoiBin
import scipy.stats
import scipy.spatial.distance

from .models import gp
from . import load


def win_prob_mat(team_skills, logistic_scale):
    """
    Compute a matrix of pairwise win probabilities based on teams' skills.

    Args:
        team_skills (pandas.Series): A series of team skills.
        logistic_scale (float): The scaling factor for a logistic win
            probability.
    """
    upper = np.triu(scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(
            team_skills,
            metric=lambda x, y: gp.logistic_win_prob(x - y, logistic_scale))))
    lower = np.tril(scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(
            team_skills,
            metric=lambda x, y: gp.logistic_win_prob(y - x, logistic_scale))))

    return pd.DataFrame(upper + lower, index=team_skills.index,
                        columns=team_skills.index)


def win_oe_pval(win_probs, outcomes):
    """
    Given predicted Bernoulli win probabilities and actual outcomes, compute
    Poisson binomial P value.

    Args:
        win_probs (numpy.ndarray): 1D array of win probabilities of each
            match.
        outcomes (numpy.ndarray): 1D binary array of match outcomes.

    Returns:
        int: Total number of matches.
        int: Observed number of wins.
        float: Expected number of wins.
        float: Poisson binomial P value for the observed count.
        float: Poisson distribution P value for the observed count.
    """
    assert len(win_probs) == len(outcomes)
    exp_wins = np.sum(win_probs)
    obs_wins = np.sum(outcomes)
    poibin_alpha = PoiBin(win_probs).cdf(obs_wins)
    if poibin_alpha < 0.5:
        poibin_pval = poibin_alpha * 2
    else:
        poibin_pval = (1 - poibin_alpha) * 2
    pois_alpha = scipy.stats.poisson.cdf(obs_wins, exp_wins)
    if pois_alpha < 0.5:
        pois_pval = pois_alpha * 2
    else:
        pois_pval = (1 - pois_alpha) * 2
    return len(win_probs), obs_wins, exp_wins, poibin_pval, pois_pval


def win_oe_pval_series(win_probs, outcomes):
    """Return a series of the win_oe_pval() output."""
    res = win_oe_pval(win_probs, outcomes)
    idx = ['nmatch', 'nwin', 'exp_win', 'poibin_pval', 'pois_pval']
    return pd.Series(res, index=idx)


class MatchPred:
    """Class for storing and analysing match predictions."""

    def __init__(self, matches, match_pred, skills_pred=None,
                 skills_pred_var=None):
        assert isinstance(matches, load.MatchDF)
        self._validate_match_pred(match_pred)

    def _validate_match_pred(self, match_pred):
        """Validate columns in the match predictions table."""
        expected_columns = [
            'startTimestamp',
            'pred_win_prob',
            'win_prob-2sd',
            'win_prob+2sd',
            'radi_skill',
            'radi_skill_sd',
            'dire_skill',
            'dire_skill_sd'
            'radi_adv']
        if not all(match_pred.index == sorted(expected_columns)):
            raise ValueError("Unexpected columns in match_red.")
        if not all(match_pred.index.isin(self.matches.df.index)):
            raise ValueError("The index of match_pred does not match that of "
                             "self.matches.df.index.")
