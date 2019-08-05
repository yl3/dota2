"""Functions for statistical analysis of the dataset."""


import numpy as np
import pandas as pd
import scipy.spatial.distance
from .models import gp


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
