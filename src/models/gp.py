"""Gaussian process modelling of player skills."""


import math
import numpy as np
import scipy.spatial.distance
import scipy.stats
import sys


def exp_cov_mat(x, scale=1.0):
    """Exponential (Ornstein-Uhlenbeck) distance covariance matrix.

    At a distance |x_i - x_j| of 1 * scale, the correlation between the two
    data points is ~0.35. At 0.25 * scale and 0.5 * scale, the correlations
    are around 0.8 and 0.6, respectively
    (see Fig. 4.1 in http://www.gaussianprocess.org/gpml/chapters/RW4.pdf).

    A good scale is could be 2 years.
    """
    condensed_d = scipy.spatial.distance.pdist(np.array(x)[:, np.newaxis],
                                               'minkowski', p=1.0)
    covar_mat = scipy.spatial.distance.squareform(
        math.exp(-condensed_d / scale))
    return covar_mat


def win_prob(skill_diff, scaling_factor=0.2):
    """Win probability for a match given a skill differential.

    Computed as 1 / (1 + math.exp(-skill_diff * scaling_factor)).

    If the skill difference of a team is 2 (i.e. 2 SDs of one player),
    and we wanted this to lead to a 60% edge, the scaling factor should be
    around 0.2.
    """
    win_prob = 1 / (1 + math.exp(skill_diff * scaling_factor))
    return win_prob


def _played_vec_to_cov_mat(cov_func, played_vec):
    return cov_func(played_vec)


def _compute_cov_mats(cov_func, time, played_mat):
    """Compute covariance matrices for each player.

    Args:
        cov_func (callable): Function from an *n* sized vector to an *n* *
            *n* size covariance matrix.
        played_mat (numpy.ndarray): A 2D boolean array of matches and
            players. The value indicates whether a player played in a game.
        time (numpy.ndarray): An 1D array of times or locations
            corresponding to each row in `played_mat`.
    """
    cov_mats = []
    for col_idx in range(played_mat.shape[1]):
        played_idx = played_mat[:, col_idx]
        cov_mats.append(_played_vec_to_cov_mat(cov_func, time[played_idx]))


def _initialise_skills(players_mat, n_cols, radiant_win):
    """Initialise skills."""
    random_skills = np.sort(np.random.normal(size=10 * players_mat.shape[0]))
    return random_skills


class SkillsGP:
    """A Gaussian process for skills.

    Args:
        players_mat (numpy.ndarray): A 2D array with shape of *M* rows for
            matches and *N* columns for players. The values are in {1, -1}
            corresponding to playing on the radiant side, the dire side or
            not at all in the respective match.
        start_times (numpy.ndarray): A 1D array of *M* match start times in
            UTC timestamp (in milliseconds).
        radiant_win (numpy.ndarray): A 1D array of *M* boolean match
            outcomes.
        player_ids (numpy.ndarray): A list of *n* player IDs.
        cov_func_name (str): The covariance function to be used. Must be in
            COV_FUNCS.
        cov_func_kwargs (dict): Keyword arguments for the covariance
            function.
        propose_sd (float): Standard deviation for the proposal Gaussian
            distribution.

    Attributes:
        M (int): Number of matches.
        N (int): Number of players.
        nan_mask (numpy.ndarray): Array of the same shape as players_mat,
            indicating whether a player did not play in a match.
        flat_nanmask (numpy.ndarray): Same as nan_mask, but flattened into
            1D.
        cov_func (callable): Covariance function mapping a vector of times
            or locations x to a covariance matrix.
        cov_mat (list): List of covariance matrices for each player whenever
            they played.
        prior_logprob (list): List of random multivariate normal variables
            with 0.0 mean and a covariance of `self.cov_mat[k]`. Log-prior
            probability for a vector `x` for the *k*'th player can be
            computed using ``self.cov_mat[k].logpdf(x)``
        skills (list): A list of 1D arrays of length 10 * *M* skills for
            each iteration. To convert into *M* * *N* matrix, use
            self.to_mat(). Skills fill the matrix row by row.
        log_posterior (list): A list of the Bayesian formula (log) numerator
            values for each corresponding vector of skills in `self.skills`.
    """

    COV_FUNCS = {
        "exponential": exp_cov_mat
    }

    def __init__(self, players_mat, start_times, radiant_win, player_ids,
                 cov_func_name, cov_func_kwargs=None, propose_sd=0.2):
        # Some basic sanity checks.
        # 10 players per game?
        # assert all(np.nansum(np.abs(players_mat), 1) == 10)
        # assert all(np.nansum(players_mat, 1) == 0)  # 5 a side?

        # Save basic data.
        self.players_mat = players_mat
        self.M, self.N = players_mat.shape
        self.start_times = start_times
        self.player_ids = player_ids
        self.propose_sd = propose_sd

        # Save computed values.
        sys.stderr.write("\rComputing self.radiant_win...\n")
        self.radiant_win = np.where(radiant_win, 1, 0)
        sys.stderr.write("\rComputing self.nanmask...\n")
        self.nanmask = players_mat == 0
        sys.stderr.write("\rComputing self.flat_nanmask...\n")
        self.flat_nanmask = self.nanmask.reshape(-1)
        sys.stderr.write("\rComputing self.cov_func...\n")
        self.cov_func = lambda x: self.COV_FUNCS[cov_func_name](
            x, **cov_func_kwargs)
        sys.stderr.write("\rComputing self.cov_mat...\n")
        self.cov_mat = _compute_cov_mats(self.cov_func, self.start_times,
                                         ~self.nanmask)
        sys.stderr.write("\rComputing self.prior_logprob...\n")
        self.prior_logprob = [scipy.stats.multivariate_normal()]

        # Initialise other variables.
        self.skills = []
        self.log_posterior = []

    def to_mat(self, vec):
        """Fill a vector of values into the slots in self.play_mask."""
        flat_mat = np.repeat(np.nan, self.M * self.N)
        flat_mat[~self.flat_nanmask] = vec
        mat = flat_mat.reshape(self.M, self.N)
        return mat

    def _initialise_skills(self):
        """Initialise skills."""
        return _initialise_skills(self.players_mat, self.N, self.radiant_win)

    def _propose_next(self):
        """Take the latest skills vector and propose the next vector."""
        if len(self.skills) == 0:
            return self._initialise_skills()
        prev_skills = self.skills[-1]
        next_skills = np.random.normal(prev_skills, self.propose_sd)
        return next_skills

    def _skills_mat_prior_logprob(self, skills_mat):
        """Gaussian process log-probability for a matrix of skills."""
        player_prior_logprobs = []
        for i in range(self.N):
            skills_vec = skills_mat[:, i]
            logprob = self.prior_logprob[i](skills_vec[~np.isnan(skills_vec)])
            player_prior_logprobs.append(logprob)
        return player_prior_logprobs

    def _match_loglik(self, skills_mat):
        """Log-likelihood of the observed outcomes."""
        match_skills_diff = -(self.players_mat * skills_mat).nansum(1)
        match_win_prob = win_prob(match_skills_diff)
        match_loglik = scipy.stats.bernoulli.logpmf(self.radiant_win,
                                                    match_win_prob)
        return match_loglik

    def iterate(self):
        """Perform a Metropolis-Hastings iteration."""
        next_skills = self._propose_next()
        next_skills_mat = self.to_mat(next_skills)
        next_prior_logprob = sum(
            self._skills_mat_prior_logprob(next_skills_mat))
        next_loglik = np.sum(self._match_loglik(next_skills_mat))
        next_logposterior = next_prior_logprob + next_loglik

        if len(self.skills) == 0:
            transition = True
        else:
            bayes_factor = math.exp(next_logposterior - self.log_posterior[-1])
            if np.random.uniform() > bayes_factor:
                transition = True
            else:
                transition = False
        if transition:
            self.skills.append(next_skills)
            self.log_posterior.append(next_logposterior)
