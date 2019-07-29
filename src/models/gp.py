"""Gaussian process modelling of player skills."""


import copy
import numpy as np
import pandas as pd
import progressbar
import scipy.linalg
import scipy.spatial.distance
import scipy.stats
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


def _dist_to_cov_mat(cov_func, played_vec):
    """Convert a vector of 1D positions to a covariance matrix."""
    return cov_func(played_vec)


def _dropna(a):
    """Drop nans from a 1D numpy array."""
    return a[~np.isnan(a)]


class GPSample:
    """A lightweight container for GP samples."""
    def __init__(self, iter, skills, radi_adv, log_posterior):
        self.iter = iter
        self.skills = skills.astype(np.float64)
        self.radi_adv = radi_adv
        self.log_posterior = log_posterior

    def __str__(self):
        return "[{}] log_posterior: {}, radi_adv: {}, skills: {}".format(
            self.iter, self.log_posterior, self.radi_adv, self.skills)


class GPSampleSet:
    """Container for storing and quering samples from a GP.

    Samples are stored as a 1D array.

    Arguments:
        players_mat (numpy.ndarray): A 2D array of 0, +1 or -1 values
            indicating whether each player (column) played for the
            radiant (+1) or dire (-1) side for each match (row).
        players (pandas.Series): A series of player names indexed by
            player ID. They are considered to be in the same order as
            the columns in `players_mat`.
        match_index (pd.Index): An indexer for matches.

    Attributes:
        N (int): Number of players.
        samples (pd.Series): A series of 1D skills vectors.
    """
    def _compute_games_of_players(self, players_mat):
        """Return a list of indices for matches in which each player played."""
        res = [np.where(players_mat.iloc[:, k] != 0)[0]
               for k in range(players_mat.shape[1])]
        return res

    def _compute_skill_vec_idx_of_player(self, match_idx_of_players):
        """
        In a flattened skills vector, return the start (0-based) and end
        coordinates of the values of the player.
        """
        idx_offsets = np.cumsum([0] + [len(x) for x in match_idx_of_players])
        return list(zip(idx_offsets[:-1], idx_offsets[1:]))

    def _skills_vec_to_mat(self, skills_vec):
        """Convert a skills vector into a skills matrix."""
        full_skills_vec = np.repeat(np.nan, self.M * self.N)
        full_skills_vec[self._played_idx] = skills_vec
        skills_mat = full_skills_vec.reshape((self.M, self.N), order='F')
        return skills_mat

    def _sample_to_radi_win_prob(self, sample):
        """Compute the per-match win probability of a sample."""
        skills_mat = self._skills_vec_to_mat(sample.skills)
        skill_diffs = (np.nansum(skills_mat * self.players_mat, 1)
                       + sample.radi_adv)
        win_prob_vec = win_prob(skill_diffs, self.win_prob_scale)
        return win_prob_vec

    def __str__(self):
        return "\n".join([str(x) for x in self.samples])

    def add_sample(self, iter, skills_by_player, radi_adv, log_posterior):
        """Add a sample.

        Arguments:
            iter (int): Current iterations.
            skills_by_player (list): A list of 1D arrays of skills per
                player.
            radi_adv (float): The radiant advantage in skills.
            log_posterior (float): The log-posterior of the current
                iteration.
            win_prob_scale (float): Scaling factor for the win
                probability function.
        """
        assert len(skills_by_player) == self.N
        concatenated_skills = np.concatenate(skills_by_player)
        assert len(concatenated_skills) == self.skills_vec_len
        assert iter not in self.samples
        self.samples[iter] = GPSample(iter, concatenated_skills, radi_adv,
                                      log_posterior)

    def skills_vec_of_kth_player(self, sample, k):
        """Skills vector of the k'th player from a sample."""
        start, end = self.skill_vec_idx_of_player[k]
        return sample.skills[start:end]

    def player_skill_by_sample(self, player, sample_slice=slice(None)):
        """Return a matrix (iteration * match) of skills."""
        if isinstance(player, int):
            idx = np.where(self.players.index == player)[0]
        elif isinstance(player, str):
            idx = np.where(self.players == player)[0]
        if len(idx) != 1:
            raise ValueError(f"Found {len(idx)} matches for {player}.")
        else:
            idx = idx[0]
        skill_by_match_mat = np.array([self.skills_vec_of_kth_player(s, idx)
                                       for s in self.samples[sample_slice]])
        iters = pd.Series([s.iter for s in self.samples[sample_slice]],
                          name='iter')
        skills_by_match_df = pd.DataFrame(
            skill_by_match_mat, index=iters,
            columns=self.match_index[self.games_by_player[idx]])
        return skills_by_match_df

    def team_skill_by_sample(self, side="radiant", sample_slice=slice(None)):
        """
        Return a matrix of (iteration * match) of total skills of a team
        (either Radiant or Dire).
        """
        if side == "radiant":
            idx = self.players_mat == 1
        elif side == "dire":
            idx = self.players_mat == -1
        else:
            raise ValueError("side must be either 'radiant' or 'dire'.")

        def skills_mat_to_team_Skill(skills_mat):
            return np.sum(np.where(idx, skills_mat, 0.0), 1)
        team_skill_mat = np.array(
            [skills_mat_to_team_Skill(self._skills_vec_to_mat(x.skills))
             for x in self.samples[sample_slice]])
        iters = pd.Series([s.iter for s in self.samples[sample_slice]],
                          name='iter')
        team_skill_df = pd.DataFrame(
            team_skill_mat, index=iters,
            columns=self.match_index)
        return team_skill_df

    def radi_adv_by_sample(self, sample_slice=slice(None)):
        """Return a series of Radiant advantages by iteration."""
        samples = self.samples[sample_slice]
        radi_advs = [s.radi_adv for s in samples]
        iters = [s.iter for s in samples]
        return pd.Series(radi_advs, index=pd.Index(iters, name="iter"))

    def radi_win_prob_by_sample(self, sample_slice=slice(None)):
        """Return a matrix (iteration * match) of Radiant win probabilities."""
        win_prob_mat = np.array([self._sample_to_radi_win_prob(s)
                                 for s in self.samples[sample_slice]])
        iters = [s.iter for s in self.samples[sample_slice]]
        win_prob_df = pd.DataFrame(
            win_prob_mat, index=pd.Index(iters, name="iter"),
            columns=self.match_index)
        return win_prob_df

    def __init__(self, players_mat, players, win_prob_scale, match_index=None):
        self.players = players
        self.players_mat = players_mat
        self.games_by_player = self._compute_games_of_players(players_mat)
        self.skill_vec_idx_of_player = \
            self._compute_skill_vec_idx_of_player(self.games_by_player)
        self.M = players_mat.shape[0]
        self.N = players_mat.shape[1]
        self.skills_vec_len = self.M * 10  # 10 players per match.
        self.samples = pd.Series([], index=pd.Int64Index([]), dtype=object)
        self._played_idx = \
            np.where(players_mat.values.reshape(-1, order='F') != 0)[0]
        self.win_prob_scale = win_prob_scale
        self.match_index = match_index

    def from_sample_set(sample_set):
        """Make a copy from a GPSampleSet."""
        newself = GPSampleSet(sample_set.players_mat, sample_set.players,
                              sample_set.win_prob_scale, sample_set.match_index)
        newself.samples = sample_set.samples.copy()
        return newself

    def copy(self):
        """Return a copied self."""
        newself = GPSampleSet(self.players_mat, self.players,
                              self.win_prob_scale, self.match_index)
        newself.samples = self.samples.copy()
        return newself

    def __getitem__(self, key):
        newself = self.copy()
        newself.samples = newself.samples[key]
        return newself


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
        sd_mat = scipy.linalg.cholesky(cov_mat, lower=True)
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
        player_ids (pd.Series): A series of player names indexed by player IDs.
        cov_func_name (str): The covariance function to be used. Must be in
            COV_FUNCS.
        cov_func_kwargs (dict): Keyword arguments for the covariance
            function.
        propose_sd (float): Standard deviation for the proposal Gaussian
            distribution.
        radi_prior_sd (float): Standard deviation for the zero-centred prior
            Gaussian distribution for Radiant advantage. Default: 1.
        radi_offset_proposal_sd (float): Proposal move standard deviation for
            the radiant advantage term. The prior is (currently) assumed
            to be flat.
        logistic_scale (float): Scaling factor for the logistic function for
            computing win probability.
        save_every_n_iter (int): Save iterations every how many iterations?
            Default: 100.
        initial_sample (GPSample): Initialisation sample.

    Attributes:
        M (int): Number of matches.
        N (int): Number of players.
        samples (list): A list of GPSample objects.
        player_skill_vecs (list): A list of GPVec objects of the current
            internal GP states.
    """

    COV_FUNCS = {
        "exponential": exp_cov_mat
    }

    def _expand_sparse_player_vec(self, arr, idx):
        """Expand an array of values, arr, into a self.M long array."""
        full_array = np.full(self.M, np.nan)
        full_array[idx] = arr
        return full_array

    def _save_cur_state(self):
        """Save the current state as a sample."""
        self.samples.add_sample(
            self._cur_iter,
            [x[0].transformed() for x in self.player_skill_vecs],
            self._cur_radi_adv, self._cur_log_posterior)

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
        radi_adv_prior_lprob = scipy.stats.norm.logpdf(
            radiant_adv, scale=self.radi_prior_sd)
        skills_mat = self.cur_skills_mat()
        match_loglik = self.match_loglik(skills_mat, radiant_adv)
        log_posterior = (radi_adv_prior_lprob + np.sum(prior_logprob)
                         + np.sum(match_loglik))
        return log_posterior

    def iterate_radiant_advantage(self):
        """Perform a Metropolis iteration on the Radiant advantage.

        Returns the (potentially) updated value as opposed to modifying
        `self` directly.
        """
        old_radi_prior_lprob = scipy.stats.norm.logpdf(
            self._cur_radi_adv, scale=self.radi_prior_sd)
        old_win_prob = win_prob(self._cur_skill_diffs, self.logistic_scale)
        old_match_loglik = np.sum(bernoulli_logpmf(
            self.radiant_win, old_win_prob))

        radi_adv_delta = np.random.normal(scale=self.radi_offset_proposal_sd)
        new_radi_prior_lprob = scipy.stats.norm.logpdf(
            self._cur_radi_adv + radi_adv_delta, scale=self.radi_prior_sd)
        new_win_prob = win_prob(
            self._cur_skill_diffs + radi_adv_delta, self.logistic_scale)
        new_match_loglik = np.sum(bernoulli_logpmf(
            self.radiant_win, new_win_prob))
        log_bayes_factor = (new_radi_prior_lprob - old_radi_prior_lprob
                            + new_match_loglik - old_match_loglik)
        if np.log(np.random.uniform()) < log_bayes_factor:
            self._cur_skill_diffs += radi_adv_delta
            self._cur_radi_adv += radi_adv_delta
            self._cur_log_posterior += log_bayes_factor

    def iterate_once_player_wise(self):
        """Perform a block-wise iteration across all players."""
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
                (self.players_mat.values[match_idx, i]
                 * player_skills_gp.transformed(skills_delta))
            new_skill_diffs = old_skill_diffs + skill_diffs_delta
            new_win_prob = win_prob(new_skill_diffs, self.logistic_scale)
            new_loglik = bernoulli_logpmf(radiant_win, new_win_prob)

            # Transition in-place?
            match_loglik_change = new_loglik - old_loglik
            log_bayes_factor = prior_lprob_change + match_loglik_change
            if np.log(np.random.uniform()) < log_bayes_factor:
                player_skills_gp.state += skills_delta
                self._cur_skill_diffs[match_idx] += skill_diffs_delta
                self._cur_log_posterior += log_bayes_factor

    def iterate(self, n=1, method="playerwise"):
        """Iterate n times."""

        for i in progressbar.progressbar(range(n)):
            # Update radiant advantage parameter.
            self.iterate_radiant_advantage()

            # Update skill parameters.
            if method == "playerwise":
                self.iterate_once_player_wise()
            else:
                raise ValueError(f"Iteration method '{method}' not recognised.")

            # Increase iteration count. Save current sample?
            self._cur_iter += 1
            if self._cur_iter % self.save_every_n_iter == 0:
                self._save_cur_state()

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
                 radi_prior_sd=1.0, radi_offset_proposal_sd=0.1,
                 logistic_scale=0.2, save_every_n_iter=100,
                 initial_sample=None):
        # Some basic sanity checks.
        # 10 players per game?
        # assert all(np.nansum(np.abs(players_mat), 1) == 10)
        # assert all(np.nansum(players_mat, 1) == 0)  # 5 a side?

        # Save basic data.
        assert isinstance(players_mat, pd.DataFrame)
        self.players_mat = players_mat
        self.M, self.N = players_mat.shape
        self.start_times = start_times
        self.player_ids = player_ids
        self.propose_sd = propose_sd
        self.radi_prior_sd = radi_prior_sd
        self.radi_offset_proposal_sd = radi_offset_proposal_sd
        self.logistic_scale = logistic_scale
        self.save_every_n_iter = save_every_n_iter
        self.samples = GPSampleSet(players_mat, player_ids, self.logistic_scale,
                                   players_mat.index)
        self.radiant_win = np.where(radiant_win, 1, 0)

        # Initialise running variables and the zero'th iteration.
        def cov_func(coords):
            return self.COV_FUNCS[cov_func_name](coords, **cov_func_kwargs)
        self.player_skill_vecs = []
        if initial_sample is not None:
            self._cur_iter = initial_sample.iter
            self._cur_radi_adv = initial_sample.radi_adv
            self._cur_log_posterior = initial_sample.log_posterior
            for k in range(self.N):
                played_matches = self.samples.games_by_player[k]
                cov_mat = _dist_to_cov_mat(cov_func,
                                           self.start_times[played_matches])
                inverse_sd_mat = scipy.linalg.inv(
                    scipy.linalg.cholesky(cov_mat, lower=True))
                transformed_skills = self.samples.skills_vec_of_kth_player(
                    initial_sample, k)
                untransformed_skills = inverse_sd_mat @ transformed_skills
                self.player_skill_vecs.append(
                    (GPVec(untransformed_skills, cov_mat),
                     played_matches))
            self._cur_skill_diffs = np.nansum(
                (self.samples._skills_vec_to_mat(initial_sample.skills)
                 * self.players_mat),
                1)
            self._cur_skill_diffs += initial_sample.radi_adv
            self.samples.samples[initial_sample.iter] = \
                copy.deepcopy(initial_sample)
        else:
            self._cur_iter = 0
            self._cur_radi_adv = 0.0
            for k in range(self.N):
                played_matches = \
                    np.arange(self.M)[self.players_mat.values[:, k] != 0.0]
                initial_values = np.repeat(0.0, len(played_matches))
                cov_mat = _dist_to_cov_mat(
                    cov_func, self.start_times[played_matches])
                self.player_skill_vecs.append((GPVec(initial_values, cov_mat),
                                               played_matches))
            self._cur_skill_diffs = \
                (np.nansum(self.players_mat * self.cur_skills_mat(), 1)
                 + self._cur_radi_adv)
            self._cur_log_posterior = self.cur_log_posterior(self._cur_radi_adv)
            self._save_cur_state()
