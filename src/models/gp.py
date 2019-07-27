"""Gaussian process modelling of player skills."""


import numpy as np
import progressbar
import scipy.linalg
import scipy.spatial.distance
import scipy.stats
import sys
import warnings


def bernoulli_logpmf(y, p):
    """Bernoulli log-pmf without checks."""
    loglik = np.sum(np.log(np.where(y == 1, p, 1 - p)))
    return loglik


def exp_cov_mat(x, scale=1.0):
    """Exponential (Ornstein-Uhlenbeck) distance covariance matrix.

    At a distance |x_i - x_j| of 1 * scale, the correlation between the two
    data points is ~0.35. At 0.25 * scale and 0.5 * scale, the correlations
    are around 0.8 and 0.6, respectively
    (see Fig. 4.1 in http://www.gaussianprocess.org/gpml/chapters/RW4.pdf).

    A good scale is could be 2 years.
    """
    squeezed_dist = scipy.spatial.distance.pdist(np.array(x)[:, np.newaxis],
                                                 'minkowski', p=1.0)
    cov_mat = scipy.spatial.distance.squareform(
        np.exp(-squeezed_dist / scale))
    cov_mat += np.diag(np.repeat(1.0, cov_mat.shape[0]))
    return cov_mat


def win_prob(skill_diff, scaling_factor):
    """Win probability for radiant given a skill difference.

    skill_diff is positive when radiant has a higher skill, i.e.
    `skill_diff` > 0 => win probability >= 50%.

    Computed as 1 / (1 + np.exp(-skill_diff / scaling_factor)).
    """
    win_prob = 1 / (1 + np.exp(-skill_diff / scaling_factor))
    return win_prob


def compute_match_win_prob(players_mat, skills_mat, radi_offset, *args):
    """Compute the win probability of each match.

    `radi_offset` is the offset advantage for the Radiant team.
    """
    match_skills_diff = np.nansum(players_mat * skills_mat, 1) + radi_offset
    match_win_prob = win_prob(match_skills_diff, *args)
    return match_win_prob


def _played_vec_to_cov_mat(cov_func, played_vec):
    return cov_func(played_vec)


def _dropna(a):
    """Drop nans from a 1D numpy array."""
    return a[~np.isnan(a)]


class GPVec():
    """
    A container for sampling a Gaussian process using a standard
    multivariate normal.

    All the data are internally stored in the standard (multivariate normal)
    space and transformed into the GP space on demand using the covariance
    matrix S.

    The GP covariance function used is exp(-|d|/s), where d is distance
    and s is the scaling factor. Rescaling s through s -> s * u is
    equivalent to exp(-d/(su)) -> exp(-d/s) ** (1/u).

    Args:
        initial_values (numpy.ndarray): Initial values in the standard
            space.
        cov_mat (numpy.ndarray): Covariance matrix.
    """
    def _validate_args(self, values, cov_mat):
        k = len(values)
        if cov_mat.shape != (k, k):
            raise ValueError("cov_mat is of the wrong shape.")

    def __init__(self, initial_values, cov_mat):
        self._validate_args(initial_values, cov_mat)
        self.state = initial_values.astype(np.longdouble)
        self.cov_mat = cov_mat
        self.cov_func_scale = 1.0

    def transformed(self, state=None, cov_func_scale=None):
        cov_mat = self.cov_mat
        if cov_func_scale is not None:
            cov_mat = cov_mat ** (1 / cov_func_scale)
        sd_mat = scipy.linalg.cholesky(cov_mat)
        if state is None:
            state = self.state
        return sd_mat @ state

    def _std_normal_logpdf(self, x):
        res = np.sum(scipy.stats.norm.logpdf(x))
        return res

    def loglik(self, delta=None, cov_func_scale=None):
        """Compute the current (transformed) log-likelihood."""
        state = self.state
        if delta is not None:
            state = np.add(state, delta)
        cov_mat = self.cov_mat
        if cov_func_scale is not None:
            cov_mat = np.power(cov_mat, (1 / cov_func_scale))
        abs_det = np.linalg.slogdet(cov_mat)[1]
        loglik = self._std_normal_logpdf(state) - 0.5 * abs_det
        return loglik

    def delta_loglik(self, delta):
        """
        Difference in log-likelihood based on a delta in the current state.

        The internal state is in standard normal. Therefore, the likelihood
        difference can be computed in a very straightforward manner.
        """
        loglik_diff = -np.sum(delta * (self.state + 0.5 * delta))
        # loglik_diff = (self._std_normal_logpdf(self.state + delta)
        #                - self._std_normal_logpdf(self.state))
        return loglik_diff


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
        radi_offset_proposal_sd (float): Proposal move standard deviation for
            the radiant advantage term. The prior is (currently) assumed
            to be flat.
        logistic_scale (float): Scaling factor for the logistic function for
            computing win probability.
        save_every_n_iter (int): Save iterations every how many iterations?
            Default: 100.

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
            computed using ``self.prior_multinorm[k](x)``
        samples (list): A list of tuples with values of (<iteration number>,
            <skills vector>, <radiant_advantage>, <log_posterior>). To
            convert a skills vector into a *M* * *N* matrix, use
            ``self.skills_vec_to_mat(skills_vec)``.
    """

    COV_FUNCS = {
        "exponential": exp_cov_mat
    }

    def _expand_sparse_player_vec(self, arr, idx):
        """Expand an array of values, arr, into a self.M long array."""
        full_array = np.full(self.M, np.nan)
        full_array[idx] = arr
        return full_array

    def prior_log_prob(self, cov_func_scale=None):
        """Total prior log-probability of the current states."""
        log_probs = [x[0].loglik(cov_func_scale=cov_func_scale)
                     for x in self.player_skill_vecs]
        return log_probs

    def cur_skills_mat(self):
        """Create the current skills matrix."""
        skills_per_player = [x[0].transformed() for x in self.player_skill_vecs]
        expanded_skills_vecs = \
            [self._expand_sparse_player_vec(skills_per_player[k],
                                            self.player_skill_vecs[k][1])
             for k in range(self.N)]
        skills_mat = np.array(expanded_skills_vecs).T
        return skills_mat

    def match_loglik(self, skills_mat, radiant_adv):
        """Compute the current log-likelihood of the observed outcomes."""
        match_win_prob = compute_match_win_prob(self.players_mat, skills_mat,
                                                radiant_adv,
                                                self.logistic_scale)
        match_loglik = bernoulli_logpmf(self.radiant_win, match_win_prob)
        return match_loglik

    def cur_log_posterior(self, radiant_adv, cov_func_scale=None):
        """Compute the log posterior probability of the skills vector."""
        prior_logprob = self.prior_log_prob(cov_func_scale)
        skills_mat = self.cur_skills_mat()
        match_loglik = self.match_loglik(skills_mat, radiant_adv)
        log_posterior = np.sum(prior_logprob) + np.sum(match_loglik)
        return log_posterior

    def get_updated_radiant_advantage(self):
        """Perform a Metropolis iteration on the Radiant advantage.

        Returns the (potentially) updated value as opposed to modifying
        `self` directly.
        """
        old_win_prob = win_prob(self._cur_skill_diffs, self.logistic_scale)
        old_match_loglik = np.sum(bernoulli_logpmf(
            self.radiant_win, old_win_prob))

        radi_adv_delta = np.random.normal(scale=self.radi_offset_proposal_sd)
        new_win_prob = win_prob(
            self._cur_skill_diffs + radi_adv_delta, self.logistic_scale)
        new_match_loglik = np.sum(bernoulli_logpmf(
            self.radiant_win, new_win_prob))
        if np.log(np.random.uniform()) < new_match_loglik - old_match_loglik:
            return (self._cur_radi_adv + radi_adv_delta,
                    self._cur_log_posterior + new_match_loglik
                    - old_match_loglik)
        else:
            return self._cur_radi_adv, self._cur_log_posterior

    def iterate_once_player_wise(self):
        """Perform a block-wise iteration across all players."""
        # First iterate on the radiant advantage offset.
        new_radi_adv, new_log_posterior = self.get_updated_radiant_advantage()
        if new_radi_adv != self._cur_radi_adv:
            self._cur_skill_diffs += new_radi_adv - self._cur_radi_adv
            self._cur_radi_adv = new_radi_adv
            self._cur_log_posterior = new_log_posterior

        # Then update each player's skills in turn.
        # transitioned_skills_mat is used for storing the updated values per
        # player.
        for i in range(self.N):
            player_skills_gp = self.player_skill_vecs[i][0]
            match_idx = self.player_skill_vecs[i][1]
            skills_delta = np.random.normal(scale=self.propose_sd,
                                            size=len(match_idx))

            # Compute the prior probability portion.
            prior_lprob_change = player_skills_gp.delta_loglik(skills_delta)

            # Compute the match likelihood portion: old match likelihood.
            old_skill_diffs = self._cur_skill_diffs[match_idx]
            radiant_win = self.radiant_win[match_idx]
            old_win_prob = win_prob(old_skill_diffs, self.logistic_scale)
            old_loglik = bernoulli_logpmf(radiant_win, old_win_prob)

            # Compute the match likelihood portion: new match likelihood.
            skill_diffs_delta = \
                (self.players_mat[match_idx, i]
                 * player_skills_gp.transformed(skills_delta))
            new_skill_diffs = old_skill_diffs + skill_diffs_delta
            new_win_prob = win_prob(new_skill_diffs, self.logistic_scale)
            new_loglik = bernoulli_logpmf(radiant_win, new_win_prob)

            # Transition in-place?
            match_loglik_change = np.sum(new_loglik) - np.sum(old_loglik)
            log_bayes_factor = prior_lprob_change + match_loglik_change
            if np.log(np.random.uniform()) < log_bayes_factor:
                player_skills_gp.state += skills_delta
                self._cur_skill_diffs[match_idx] += skill_diffs_delta
                self._cur_log_posterior += log_bayes_factor

        # Increase iteration count. Save current sample?
        self._cur_iter += 1
        if self._cur_iter % self.save_every_n_iter == 0:
            cur_states = [x[0].state.copy().astype(np.float64)
                          for x in self.player_skill_vecs]
            self.samples.append((self._cur_iter, cur_states, self._cur_radi_adv,
                                 self._cur_log_posterior))

    def iterate(self, n=1, method="playerwise"):
        """Iterate n times."""

        if method == "playerwise":
            for i in progressbar.progressbar(range(n)):
                try:
                    self.iterate_once_player_wise()
                except KeyboardInterrupt:
                    sys.stderr.write(f"Interrupted at iteration {i}.\n")
                    break
        else:
            raise ValueError(f"Iteration method '{method}' not recognised.")

        # Check the integrity of the running sums.
        if not np.allclose(
                (np.nansum(self.players_mat * self.cur_skills_mat(), 1)
                 + self._cur_radi_adv),
                self._cur_skill_diffs):
            warnings.warn("Loss of integrity with self._cur_skill_diffs")
        if not np.isclose(self.cur_log_posterior(self._cur_radi_adv),
                          self._cur_log_posterior):
            warnings.warn("Loss of integrity with self._cur_log_posterior")

    def __init__(self, players_mat, start_times, radiant_win, player_ids,
                 cov_func_name, cov_func_kwargs=None, propose_sd=0.2,
                 radi_offset_proposal_sd=0.005, logistic_scale=0.2,
                 save_every_n_iter=100):
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
        self.radi_offset_proposal_sd = radi_offset_proposal_sd
        self.logistic_scale = logistic_scale
        self.save_every_n_iter = save_every_n_iter

        # Save computed values.
        self.radiant_win = np.where(radiant_win, 1, 0)
        self.cov_func = lambda x: self.COV_FUNCS[cov_func_name](
            x, **cov_func_kwargs)
        self.player_skill_vecs = []
        for k in range(self.N):
            played_matches = np.arange(self.M)[self.players_mat[:, k] != 0.0]
            initial_values = np.repeat(0.0, len(played_matches))
            cov_mat = _played_vec_to_cov_mat(self.cov_func,
                                             self.start_times[played_matches])
            self.player_skill_vecs.append((GPVec(initial_values, cov_mat),
                                           played_matches))

        # Initialise other variables:
        # Current skill differences at each match.
        self._cur_iter = 0
        self._cur_radi_adv = 0.0
        self._cur_skill_diffs = \
            (np.nansum(self.players_mat * self.cur_skills_mat(), 1)
             + self._cur_radi_adv)
        self._cur_log_posterior = self.cur_log_posterior(self._cur_radi_adv)
        self.samples = [(self._cur_iter,
                        [x[0].state for x in self.player_skill_vecs],
                        self._cur_radi_adv, self._cur_log_posterior)]
