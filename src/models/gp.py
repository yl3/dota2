"""Gaussian process modelling of player skills."""


import numpy as np
import progressbar
import scipy.linalg
import scipy.spatial.distance
import scipy.stats


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
    covar_mat = scipy.spatial.distance.squareform(
        np.exp(-squeezed_dist / scale))
    covar_mat += np.diag(np.repeat(1.0, covar_mat.shape[0]))
    return covar_mat


def win_prob(skill_diff, scaling_factor=0.2):
    """Win probability for radiant given a skill difference.

    skill_diff is positive when radiant has a higher skill, i.e.
    `skill_diff` > 0 => win probability >= 50%.

    Computed as 1 / (1 + np.exp(-skill_diff * scaling_factor)).

    If the skill difference of a team is 2 (i.e. 2 SDs of one player),
    and we wanted this to lead to a 60% edge, the scaling factor should be
    around 0.2.

    The value of 0.2 also corresponds to a difference of 10 points
    resulting in a win rate of 85-90%.
    """
    win_prob = 1 / (1 + np.exp(-skill_diff * scaling_factor))
    return win_prob


def compute_match_win_prob(players_mat, skills_mat):
    """Compute the win probability of each match."""
    match_skills_diff = np.nansum(players_mat * skills_mat, 1)
    match_win_prob = win_prob(match_skills_diff)
    return match_win_prob


def _played_vec_to_cov_mat(cov_func, played_vec):
    return cov_func(played_vec)


def _compute_cov_mats(cov_func, time, played_mat, delta=1e-9):
    """Compute covariance matrices for each player.

    Args:
        cov_func (callable): Function from an *n* sized vector to an *n* *
            *n* size covariance matrix.
        played_mat (numpy.ndarray): A 2D boolean array of matches and
            players. The value indicates whether a player played in a game.
        time (numpy.ndarray): An 1D array of times or locations
            corresponding to each row in `played_mat`.
        delta (float): Add a small delta to make sure the matrix if positive
            semidefinite.
    """
    cov_mats = []
    for col_idx in range(played_mat.shape[1]):
        played_idx = played_mat[:, col_idx]
        cov_mat = _played_vec_to_cov_mat(cov_func, time[played_idx])
        cov_mat += np.diag(np.repeat(delta, cov_mat.shape[0]))
        cov_mats.append(cov_mat)
    return cov_mats


def _initialise_skills(players_mat, n_cols, radiant_win):
    """Initialise skills."""
    random_skills = np.random.normal(size=10 * players_mat.shape[0])
    return random_skills


def _c_to_r_idx(bool_mat):
    """
    Compute index for converting a 1D representation of a sparse matrix from
    column by column ordering to row by row ordering.

    If ``x`` is an array of values corresponding to True values in bool_mat
    when reshaped row by row (default in numpy), then ``x[idx]`` (where
    `idx` is the return value of this function) reorders the values in ``x``
    such that they correspond to the matrix when reshaped column by column.

    Args:
        bool_mat (numpy.ndarray): A boolean matrix of indicating which
            values are represented by the input vector.

    Returns:
        numpy.ndarray: A 1D array of indices.
    """
    # Algorithm: create an increasing index array and reorder it by
    # converting to and from a matrix.
    # Initialise an index array with nans.
    idx = np.repeat(-1, bool_mat.shape[0] * bool_mat.shape[1])

    # Treat the index array as a column-by-column format and fill in the
    # index values.
    idx[bool_mat.reshape(-1, order='F')] = np.arange(np.sum(bool_mat))

    # Reshape the index array column-wise into a matrix then row-wise back
    # to a 1D array.
    idx = idx.reshape(bool_mat.shape, order='F').reshape(-1)
    idx = idx[idx != -1]

    return idx


def _dropna(a):
    """Drop nans from a 1D numpy array."""
    return a[~np.isnan(a)]


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
            <skills vector>, <log_posterior>). To convert a skills vector
            into a *M* * *N* matrix, use
            ``self.skills_vec_to_mat(skills_vec)``.
    """

    COV_FUNCS = {
        "exponential": exp_cov_mat
    }

    def _initialise_skills(self):
        """Initialise skills.

        Returns:
            tuple: A sample compatible with :attr:`samples`.
        """
        # return _initialise_skills(self.players_mat, self.N, self.radiant_win)
        initial_skills = np.repeat(0.0, self.M * 10)
        log_prob = self.compute_log_posterior(initial_skills)
        return (0, initial_skills, log_prob)

    def _propose_move_for_player(self, k):
        """Propose a move for a player.

        I.e. for column `k` in self.player_mat.
        """
        cov_mat = self.cov_mat[k]
        move = np.random.multivariate_normal(np.repeat(0.0, cov_mat.shape[0]),
                                             cov_mat * (self.propose_sd ** 2))
        return move

    def _propose_next(self):
        """Take the latest skills vector and propose the next vector."""
        prev_skills = self._cur_skills
        next_moves = [self._propose_move_for_player(k) for k in range(self.N)]
        next_moves = np.concatenate(next_moves)
        # next_moves is now in column-to-column order. Change that into
        # row-by-row order.
        next_moves = next_moves[self._c_to_r_idx]
        next_skills = prev_skills + next_moves
        return next_skills

    def _skills_mat_prior_logprob(self, skills_mat):
        """Gaussian process log-probability for a matrix of skills."""
        player_prior_logprobs = []
        for i in range(self.N):
            skills_vec = skills_mat[:, i]
            skills_vec_f = _dropna(skills_vec)
            logprob = self.prior_multinorm[i](skills_vec_f)
            player_prior_logprobs.append(logprob)
        return player_prior_logprobs

    def _match_loglik(self, skills_mat):
        """Log-likelihood of the observed outcomes."""
        match_win_prob = compute_match_win_prob(self.players_mat, skills_mat)
        match_loglik = scipy.stats.bernoulli.logpmf(self.radiant_win,
                                                    match_win_prob)
        return match_loglik

    def skills_vec_to_mat(self, vec):
        """
        Fill a vector of values into the slots in self.play_mask row by row.

        Args:
            vec (numpy.ndarray): 1D array of length self.M * 10. Default:
                self._cur_skills
        """
        flat_mat = np.repeat(np.nan, self.M * self.N)
        flat_mat[~self.flat_nanmask] = vec
        mat = flat_mat.reshape(self.M, self.N)
        return mat

    def compute_log_posterior(self, skills_vec):
        """Compute the log posterior probability of the skills vector."""
        assert len(skills_vec) == self.M * 10
        skills_mat = self.skills_vec_to_mat(skills_vec)
        prior_logprob = sum(
            self._skills_mat_prior_logprob(skills_mat))
        loglik = np.sum(self._match_loglik(skills_mat))
        log_posterior = prior_logprob + loglik
        return log_posterior

    def iterate_once_player_wise(self):
        """Perform a block-wise iteration across all players."""
        old_skills_mat = self.skills_vec_to_mat(self._cur_skills)
        transitioned_skills_mat = old_skills_mat.copy()
        for i in range(self.N):
            # Compute the prior probability portion.
            next_move = self._propose_move_for_player(i)
            old_skills_vec = _dropna(old_skills_mat[:, i])
            new_skills_vec = old_skills_vec + next_move
            old_prior_lprob = self.prior_multinorm[i](old_skills_vec)
            new_prior_lprob = self.prior_multinorm[i](new_skills_vec)

            # Compute the match likelihood portion.
            affected_matches = ~self.nanmask[:, i]
            old_match_win_prob = compute_match_win_prob(
                self.players_mat[affected_matches, :],
                old_skills_mat[affected_matches, :])
            new_skills_mat = transitioned_skills_mat[affected_matches, :]
            new_skills_mat[:, i] = new_skills_vec
            old_match_loglik = scipy.stats.bernoulli.logpmf(
                self.radiant_win[affected_matches],
                old_match_win_prob)
            new_match_win_prob = compute_match_win_prob(
                self.players_mat[affected_matches, :], new_skills_mat)
            new_match_loglik = scipy.stats.bernoulli.logpmf(
                self.radiant_win[affected_matches],
                new_match_win_prob)

            # Transition?
            log_bayes_factor = (new_prior_lprob - old_prior_lprob
                                + sum(new_match_loglik) - sum(old_match_loglik))
            if np.log(np.random.uniform()) < log_bayes_factor:
                idx = ~self.nanmask[:, i]
                transitioned_skills_mat[idx, i] = new_skills_vec
        transitioned_skills_vec = _dropna(transitioned_skills_mat.reshape(-1))
        # TODO
        # proposed_log_posterior = self.compute_log_posterior(transitioned_skills_vec)
        assert len(transitioned_skills_vec) == self.M * 10
        prior_logprob = sum(
            self._skills_mat_prior_logprob(transitioned_skills_mat))
        loglik = np.sum(self._match_loglik(transitioned_skills_mat))
        new_log_posterior = prior_logprob + loglik

        # self._cur_skills won't change, if transitioned_skills_mat doesn't
        # change, which happens when no player's vector got updated.
        self._cur_skills = transitioned_skills_vec
        self._cur_log_posterior = new_log_posterior
        self._cur_iter += 1

        # Save the current iteration as a sample?
        if self._cur_iter % self.save_every_n_iter == 0:
            self.samples.append((self._cur_iter, self._cur_skills,
                                 self._cur_log_posterior, prior_logprob,
                                 loglik))

    def iterate_once_full(self):
        """Perform a full Metropolis iteration."""
        proposed_skills = self._propose_next()
        # TODO
        # proposed_log_posterior = self.compute_log_posterior(proposed_skills)
        assert len(proposed_skills) == self.M * 10
        skills_mat = self.skills_vec_to_mat(proposed_skills)
        prior_logprob = sum(
            self._skills_mat_prior_logprob(skills_mat))
        loglik = np.sum(self._match_loglik(skills_mat))
        proposed_log_posterior = prior_logprob + loglik

        # Transition to a new state?
        log_bayes_factor = proposed_log_posterior - self._cur_log_posterior
        if np.log(np.random.uniform()) < log_bayes_factor:
            self._cur_skills = proposed_skills
            self._cur_log_posterior = proposed_log_posterior
        self._cur_iter += 1

        # Save the current iteration as a sample?
        if self._cur_iter % self.save_every_n_iter == 0:
            self.samples.append((self._cur_iter, self._cur_skills,
                                 self._cur_log_posterior, prior_logprob,
                                 loglik))

    def iterate(self, n=1, method="full"):
        """Iterate n times."""
        if method == "full":
            for i in progressbar.progressbar(range(n)):
                self.iterate_once_full()
        elif method == "playerwise":
            for i in progressbar.progressbar(range(n)):
                self.iterate_once_player_wise()
        else:
            raise ValueError(f"Iteration method '{method}' not recognised.")

    def __init__(self, players_mat, start_times, radiant_win, player_ids,
                 cov_func_name, cov_func_kwargs=None, propose_sd=0.2,
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
        self.save_every_n_iter = save_every_n_iter

        # Save computed values.
        self.radiant_win = np.where(radiant_win, 1, 0)
        self.nanmask = players_mat == 0
        self.flat_nanmask = self.nanmask.reshape(-1)
        self._c_to_r_idx = _c_to_r_idx(~self.nanmask)
        self.cov_func = lambda x: self.COV_FUNCS[cov_func_name](
            x, **cov_func_kwargs)
        self.cov_mat = _compute_cov_mats(self.cov_func, self.start_times,
                                         ~self.nanmask)
        self.prior_multinorm = []
        for i in range(len(self.cov_mat)):
            self.prior_multinorm.append(
                scipy.stats.multivariate_normal(cov=self.cov_mat[i]).logpdf)

        # Initialise other variables.
        temp_tuple = self._initialise_skills()
        self._cur_iter, self._cur_skills, self.cur_log_posterior = temp_tuple
        self.samples = [temp_tuple]
