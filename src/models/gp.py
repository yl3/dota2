"""Gaussian process modelling of player skills."""


import copy
import math
import numpy as np
import pandas as pd
import progressbar
import scipy.linalg
import scipy.optimize
import scipy.spatial.distance
import scipy.stats
import warnings


def bernoulli_logpmf(y, p):
    """Bernoulli log-pmf without checks."""
    loglik = np.sum(np.log(np.where(y == 1, p, 1 - p)))
    return loglik


def exp_cov_mat(x1, scale, x2=None):
    """Exponential (Ornstein-Uhlenbeck) distance covariance matrix.

    At a distance |x_i - x_j| of 1 * scale, the correlation between the two
    data points is ~0.35. At 0.25 * scale and 0.5 * scale, the correlations
    are around 0.8 and 0.6, respectively
    (see Fig. 4.1 in http://www.gaussianprocess.org/gpml/chapters/RW4.pdf).

    A good scale is could be 2 years.
    """
    if x2 is None:
        x2 = x1
    cov_mat = scipy.spatial.distance.cdist(
        x1.reshape(-1, 1), x2.reshape(-1, 1), 'minkowski', p=1.0)
    cov_mat = np.exp(-cov_mat / scale)
    return cov_mat


def gp_predict(x_known, y_known, x_new, cov_func):
    """Predict new values in a GP given observed values.

    Notation used is that of
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    """
    sigma = cov_func(np.append(x_new, x_known))
    sigma_11 = sigma[:len(x_new), :len(x_new)]
    sigma_12 = sigma[:len(x_new), len(x_new):]
    sigma_21 = sigma[len(x_new):, :len(x_new)]
    sigma_22 = sigma[len(x_new):, len(x_new):]
    sigma_22_inv = scipy.linalg.inv(sigma_22)
    x_new_mu = sigma_12 @ sigma_22_inv @ y_known
    x_new_sigma = sigma_11 - sigma_12 @ sigma_22_inv @ sigma_21
    return x_new_mu, x_new_sigma


def logistic_win_prob(skill_diff, scaling_factor):
    """Win probability for Radiant given a skill difference.

    skill_diff is positive when radiant has a higher skill, i.e.
    `skill_diff` > 0 => win probability >= 50%.

    Computed as 1 / (1 + np.exp(-skill_diff / scaling_factor)).
    """
    win_prob = 1 / (1 + np.exp(-skill_diff / scaling_factor))
    return win_prob


def _dropna(a):
    """Drop nans from a 1D numpy array."""
    return a[~np.isnan(a)]


def _last_value(s, default):
    """Return the last non-nan value in series S."""
    idx = s.last_valid_index()
    if idx is None:
        return default
    else:
        return s[idx]


class SkillsGP:
    """A Gaussian process super class for skills.

    The prior mean for the GP is assumed to be 0.0, and the prior variance
    (the diagonal of the covariance matrix) is assumed to be 1.0.

    Args:
        players_mat (numpy.ndarray): A 2D array with shape of *M* rows for
            matches and *N* columns for players. The values are in {1, -1}
            corresponding to playing on the radiant side, the dire side or
            not at all in the respective match. If a player did not play a
            match, the value should be 0.0.
        start_times (numpy.ndarray): A 1D array of *M* match start times in
            UTC timestamp (in milliseconds).
        radiant_win (numpy.ndarray): A 1D array of *M* boolean match
            outcomes.
        player_ids (pd.Series): A series of player names indexed by player IDs.
        cov_func_name (str): The covariance function to be used. Must be in
            COV_FUNCS.
        cov_func_kwargs (dict): Keyword arguments for the covariance
            function.
        logistic_scale (float): Scaling factor for the logistic function for
            computing win probability.

    Attributes:
        M (int): Number of matches.
        N (int): Number of players.
    """

    COV_FUNCS = {
        "exponential": exp_cov_mat
    }

    def __init__(self, players_mat, start_times, radiant_win, cov_func_name,
                 cov_func_kwargs=None, radi_prior_sd=1.0, logistic_scale=0.2):
        # Some basic sanity checks.
        assert all(np.nansum(np.abs(players_mat), 1) == 10)
        assert all(np.nansum(players_mat, 1) == 0)  # 5 a side?
        assert isinstance(players_mat, pd.DataFrame)
        assert all(start_times == sorted(start_times))

        # Save basic data.
        self.players_mat = players_mat
        self.M, self.N = players_mat.shape
        self.start_times = start_times
        self.radiant_win = np.where(radiant_win, 1, 0)
        self.cov_func_name = cov_func_name
        self.cov_func_kwargs = cov_func_kwargs
        self.radi_prior_sd = radi_prior_sd
        self.logistic_scale = logistic_scale

        def cov_func(coords):
            return self.COV_FUNCS[cov_func_name](coords, **cov_func_kwargs)
        self.cov_func = cov_func

        # Pre-compute values.
        self._games_by_player = self._compute_games_of_players(self.players_mat)
        self._ngames_by_player = [len(x) for x in self._games_by_player]
        # Match index corresponding to each skill index.
        self._match_of_skill_idx = self._match_of_skill_idx()
        # Side of each element in skill vector.
        self._sign_of_skill_idx = self._sign_of_skill_idx()
        # Start and end index of skills corresponding to each player.
        self._skill_slice_of_player = self._skill_slice_of_player()
        # self._skills_by_match_order can be used to reorder a skills or
        # sign vector by match.
        self._skills_by_match_order = np.argsort(self._match_of_skill_idx)
        self.cov_mats = [self.cov_func(self.start_times[played_matches])
                         for played_matches in self._games_by_player]

    def win_prob(self, skill_diffs):
        return logistic_win_prob(skill_diffs, self.logistic_scale)

    def skills_vec_to_skills_by_player(self, skills_vec):
        """
        Convert a length self.M * 10 skills vector into a list of skills by
        player.
        """
        if not hasattr(self, "_skills_vec_split_idx"):
            self._skills_vec_split_idx = np.cumsum(self._ngames_by_player[:-1])
        return np.split(skills_vec, self._skills_vec_split_idx)

    def match_skill_diffs(self, skills_vec):
        """
        Skill difference in each match, without considering Radiant advantage.
        """
        signed_skills = skills_vec * self._sign_of_skill_idx
        skill_diffs = np.sum(
            signed_skills[self._skills_by_match_order].reshape(self.M, 10),
            1)
        return skill_diffs

    def radi_win_probs(self, skills_vec, radi_adv):
        """Radiant win probability of a skills vector."""
        skill_diffs = self.match_skill_diffs(skills_vec) + radi_adv
        win_probs = self.win_prob(skill_diffs)
        return win_probs

    def match_team_skills(self, skills_vec, side="radiant"):
        """
        Skills of a team in each match, without considering Radiant advantage.
        """
        assert side in ["radiant", "dire"]
        signs_by_match = self._sign_of_skill_idx[self._skills_by_match_order]
        if side == "radiant":
            team_signed_skills = skills_vec[self._skills_by_match_order][
                signs_by_match == 1.0]
            skill_sums = np.sum(team_signed_skills.reshape(self.M, 5), 1)
        elif side == "dire":
            team_signed_skills = skills_vec[self._skills_by_match_order][
                signs_by_match == -1.0]
            skill_sums = np.sum(team_signed_skills.reshape(self.M, 5), 1)
        return skill_sums

    def match_loglik(self, skills_vec, radi_adv):
        """Compute the current log-likelihood of the observed outcomes."""
        skill_diffs = self.match_skill_diffs(skills_vec)
        match_win_prob = self.win_prob(skill_diffs + radi_adv)
        match_loglik = bernoulli_logpmf(self.radiant_win, match_win_prob)
        return match_loglik

    def log_posterior(self, skills_vec, radi_adv, inv_sd_mats=None,
                      abs_log_det=None):
        """Compute the log-posterior of a skills vector.

        Args:
            skills_vec (numpy.ndarray): 1D flattened array of skills for each
                player in each match.
            radi_adv (float): Radiant advantage.
            inv_sd_mats (list): A list of matrices A^-1, such that AAt is the
                covariance matrix of each player. By default calculated from
                self.cov_mats.
            abs_log_det (float): The absolute log-determinant of the  (full
                block diagonal) covariance matrix. By default calculate from
                self.cov_mats.
        """
        if inv_sd_mats is None:
            inv_sd_mats = \
                [scipy.linalg.inv(scipy.linalg.cholesky(x, lower=True))
                 for x in self.cov_mats]
        if abs_log_det is None:
            abs_log_dets = [np.linalg.slogdet(x)[1] for x in self.cov_mats]
            abs_log_det = np.sum(abs_log_dets)

        # Compute the skills prior log-probabilities.
        skills_of_player = self.skills_vec_to_skills_by_player(skills_vec)
        stdnorm_prior_lprobs = []
        for k in range(self.N):
            x = inv_sd_mats[k] @ skills_of_player[k]
            stdnorm_loglik = np.sum(scipy.stats.norm.logpdf(x))
            stdnorm_prior_lprobs.append(stdnorm_loglik)
        total_skill_prior_lprob = (np.sum(stdnorm_prior_lprobs)
                                   - 0.5 * abs_log_det)

        # Compute the prior probability for the Radiant advantage.
        radi_adv_lprob = scipy.stats.norm.logpdf(radi_adv,
                                                 scale=self.radi_prior_sd)

        # Compute the match log-likelihoods.
        match_loglik = self.match_loglik(skills_vec, radi_adv)
        total_loglik = total_skill_prior_lprob + radi_adv_lprob + match_loglik
        return total_loglik

    def skills_mat(self, skills_vec):
        """Convert the current skills vector into a skills matrix."""
        skills_mat_t = np.full((self.N, self.M), np.nan)
        skills_mat_t[self.players_mat.T != 0.0] = skills_vec
        skills_df = pd.DataFrame(skills_mat_t.T, index=self.players_mat.index,
                                 columns=self.players_mat.columns)
        return skills_df

    def _compute_games_of_players(self, players_mat):
        """Return a list of indices for matches in which each player played."""
        res = [np.where(players_mat.iloc[:, k] != 0)[0] for k in range(self.N)]
        return res

    def _expand_sparse_player_vec(self, arr, idx):
        """Expand an array of values, arr, into a self.M long array."""
        full_array = np.full(self.M, np.nan)
        full_array[idx] = arr
        return full_array

    def _match_of_skill_idx(self):
        """
        Return index vector of length self.M * 10, where each value m_i in
        range(self.M) corresponds to match m_i where skill i is involved in.
        """
        match_of_skill_idx = np.concatenate(
            [np.arange(self.M)[self.players_mat.values[:, k] != 0]
             for k in range(self.N)])
        return match_of_skill_idx

    def _sign_of_skill_idx(self):
        """
        Compute the sign vector of +1 or -1 for each value in the skills vector.
        """
        return self.players_mat.values.T[self.players_mat.values.T != 0.0]

    def _skill_idx_of_match(self):
        """
        Compute an index vector of length self.M, where each value a_m is in
        range(self.M * 10) and corresponds to the indices i of the skills that
        are part of match m. The skills vector "fill" the skills matrix column
        by column.
        """
        skill_idx_mat = np.full(self.players_mat.shape, -1).T
        skill_idx_mat[self.players_mat.values.T != 0.0] = np.arange(self.M * 10)
        skill_idx_of_match = [skill_idx_mat[skill_idx_mat[:, k] != -1, k]
                              for k in range(self.M)]
        return skill_idx_of_match

    def _skill_slice_of_player(self):
        """Compute the index of skills in a skill vector for the k'th player."""
        idx_offsets = np.cumsum([0] + self._ngames_by_player)
        return [slice(start, end)
                for start, end in zip(idx_offsets[:-1], idx_offsets[1:])]


class SkillsGPMAP(SkillsGP):
    """Class for computing the MAP of a skills GP model.

    Args:
        initial_values (numpy.ndarray): The initial 1D skills vector.
    """

    def __init__(self, initial_skills=None, initial_radi_adv=None, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if initial_skills is not None:
            assert len(initial_skills) == self.M * 10
            self._initial_skills = initial_skills
        else:
            self._initial_skills = np.full(self.M * 10, 0.0)
        if initial_radi_adv is not None:
            self._initial_radi_adv = initial_radi_adv
        else:
            self._initial_radi_adv = 0.0
        self.fitted = None

    def fit(self, initial_skills=None, initial_radi_adv=0.0):
        """Perform a Newton-Raphson fit of the model.

        Args:
            initial_skills (numpy.ndarray): A 1D array of skills of each player
                in each match. Default: every player's skills start from 0.0.
            initial_radi_adv (float): Initial Radiant advantage.

        Returns:
            tuple: A tuple of fitted skills and fitted Radiant advantage.
        """
        # Compute values used by minus_full_loglik() at each iteration.
        inv_sd_mats = [scipy.linalg.inv(scipy.linalg.cholesky(x, lower=True))
                       for x in self.cov_mats]
        abs_log_dets = [np.linalg.slogdet(x)[1] for x in self.cov_mats]
        abs_log_det = np.sum(abs_log_dets)

        # Compute the sparse inverse covariance matrix used by minus_gradient()
        # and minus_hessp() at each iteration.
        minus_inv_cov_mats = [-scipy.linalg.inv(x) for x in self.cov_mats]
        radi_adv_cov_mat = np.full((1, 1), -1 / (self.radi_prior_sd ** 2))
        minus_inv_cov_mat = scipy.sparse.block_diag(
            minus_inv_cov_mats + [radi_adv_cov_mat],
            format='csr')

        def minus_full_loglik(params):
            skills, radi_adv = np.split(params, [self.M * 10])
            return -self.log_posterior(skills, radi_adv, inv_sd_mats,
                                       abs_log_det)

        def minus_gradient(params):
            skills, radi_adv = params[:-1], params[-1]
            return -self._gradient(skills, radi_adv, minus_inv_cov_mat)

        def minus_hessp(params, p):
            skills, radi_adv = params[:-1], params[-1]
            return -self._hessp(skills, radi_adv, p, minus_inv_cov_mat)

        initial_values = np.append(self._initial_skills, self._initial_radi_adv)
        self.fitted = scipy.optimize.minimize(
            minus_full_loglik, initial_values, method='Newton-CG',
            jac=minus_gradient, hessp=minus_hessp)
        print(self.fitted)

    def predict(self, new_times, player_ids, as_df=True):
        """Predict skills at a new timepoint.

        Args:
            new_times (numpy.ndarray): A 1D list of new times in the same scale
                as `self.start_times`.
            player_ids (iterable): An iterable of player IDs.
            as_df (bool): Whether to return the results as a matrix or as a
                Pandas data frame.

        Returns:
            pred_mat (numpy.ndarray): A len(new_times) by len(player_ids) matrix
                or Pandas data frame of player skills.
            var_mat (numpy.ndarray): A len(new_times) by len(player_ids) matrix
                or Pandas data frame of player skill variances.
        """
        if self.fitted is None:
            raise AttributeError("Please fit the model using self.fit() first.")

        mu_of_player = []
        cov_of_player = []
        for player_id in player_ids:
            if player_id in self.players_mat.columns:
                idx = np.where(self.players_mat.columns == player_id)[0][0]
                games_idx = self._games_by_player[idx]
                x_train = self.start_times[games_idx]
                y_train = self.fitted.x[self._skill_slice_of_player[idx]]
                x_pred = new_times
                mu, cov_mat = gp_predict(x_train, y_train, x_pred,
                                         self.cov_func)
                mu_of_player.append(mu)
                cov_of_player.append(np.diagonal(cov_mat))
            else:
                # Just use the hard-coded prior.
                mu_of_player.append(0.0)
                cov_of_player.append([1.0])
        pred_mat = np.array(mu_of_player).T
        var_mat = np.array(cov_of_player).T
        if as_df:
            pred_mat = pd.DataFrame(pred_mat, index=new_times,
                                    columns=player_ids)
            var_mat = pd.DataFrame(var_mat, index=new_times, columns=player_ids)
        return pred_mat, var_mat

    def predict_matches(self, radiant_players, dire_players, start_times):
        """Predict the outcome of matches.

        Useful for backtesting.

        Args:
            radiant_players (pandas.Series): A list of lists of Radiant player
                IDs for the new matches.
            dire_players (pandas.Series): A list of lists of Dire player ID for
                the new matchess.
            start_times (array-like): A 1D integer array of time stamps in the
                same unit as `self.start_times` for the new matches.

        Returns:
            pred: A data frame of predicted Radiant team win probability, 2.5%
                win probability confidence interval, 97.5% win probability
                confidence interval, predicted Radiant skill, Radiant skill
                standard deviation, predicted Dire skill, Dire skill standard
                deviation, predicted Radiant advantage.
        """
        player_ids = radiant_players + dire_players
        predicted_results = []
        for players, start_time in zip(player_ids, start_times):
            mu_mat, var_mat = self.predict([start_time], players, as_df=False)
            radi_adv = self.fitted.x[-1]
            radi_skill = np.sum(mu_mat[:, :5])
            radi_skill_sd = math.sqrt(np.sum(var_mat[:, :5]))
            dire_skill = np.sum(mu_mat[:, 5:])
            dire_skill_sd = math.sqrt(np.sum(var_mat[:, 5:]))
            skill_diff = radi_skill - dire_skill + radi_adv
            skill_diff_sd = math.sqrt(np.sum(var_mat))
            pred_win_prob = self.win_prob(skill_diff)
            win_prob_low_bound = self.win_prob(skill_diff - 2 * skill_diff_sd)
            win_prob_high_bound = self.win_prob(skill_diff + 2 * skill_diff_sd)
            predicted_results.append((pred_win_prob, win_prob_low_bound,
                                      win_prob_high_bound, radi_skill,
                                      radi_skill_sd, dire_skill, dire_skill_sd,
                                      radi_adv))
        columns = ["pred_win_prob", "win_prob-2sd", "win_prob+2sd",
                   "radi_skill", "radi_skill_sd", "dire_skill", "dire_skill_sd",
                   "radi_adv"]
        pred_df = pd.DataFrame(predicted_results, columns=columns,
                               index=start_times)
        return pred_df

    def add_matches(self, players_mat, start_times, radiant_win):
        """Add new matches to the current fitted `SkillsGPMAP` object.

        Impute the initial skills based on the currently fitted results.
        """
        concatenated_times = np.concatenate([self.start_times, start_times])
        assert all(concatenated_times == sorted(concatenated_times))

        # A (ultimately row) vector of imputed skills per player.
        imputed_skills_of_player = []
        for player_id in players_mat.columns:
            if player_id not in self.players_mat.columns:
                # If a player doesn't have a played match yet, use the hard-
                # code default.
                imputed_skills_of_player.append(0.0)
            else:
                idx = np.where(player_id == self.players_mat.columns)[0][0]
                imputed_skills_of_player.append(
                    self.fitted.x[self._skill_slice_of_player[idx]][-1])
        imputed_skills_of_player = \
            np.array(imputed_skills_of_player).reshape(1, -1)

        # Fill the new skills matrix using the imputed skills.
        imputed_skills_mat = players_mat.copy()
        imputed_skills_mat[imputed_skills_mat == 0.0] = np.nan
        # With broadcasting, we can now fill in the non-nan values of each
        # row using the imputed skills.
        imputed_skills_mat = imputed_skills_mat * 0.0 + imputed_skills_of_player

        # Create the extended player and skills matrices.
        new_players_mat = self.players_mat.append(players_mat).fillna(0.0)
        fitted_skills_mat = self.skills_mat(self.fitted.x[:-1])
        new_skills_mat = fitted_skills_mat.append(
            pd.DataFrame(imputed_skills_mat, index=players_mat.index,
                         columns=players_mat.columns))
        assert (all(new_players_mat.index == new_skills_mat.index)
                and all(new_players_mat.columns == new_skills_mat.columns))

        # Create the new object.
        new_initial_skills = new_skills_mat.values.T[
            new_players_mat.values.T != 0.0]
        new_skillsgp = SkillsGPMAP(
            initial_skills=new_initial_skills,
            initial_radi_adv=self.fitted.x[-1],
            players_mat=new_players_mat,
            start_times=concatenated_times,
            radiant_win=np.concatenate([self.radiant_win, radiant_win]),
            cov_func_name=self.cov_func_name,
            cov_func_kwargs=self.cov_func_kwargs,
            radi_prior_sd=self.radi_prior_sd,
            logistic_scale=self.logistic_scale
        )
        return new_skillsgp

    def _gradient(self, skills, radi_adv, minus_inv_cov_mat):
        """Gradient of the model - helper function needed by self.fit().

        Argument `minus_inv_cov_mat` is a sparse block diagonal matrix of the
        inverted covariance (sub)matrices including the covariance (matrix) for
        the Radiant advantage term. I.e. the length of `minus_inv_cov_mat` is
        len(skills) + 1.
        """
        # Compute the skills prior term of the gradient.
        prior_lprob_gradient = minus_inv_cov_mat @ np.append(skills, radi_adv)

        # Compute the match log-likelihood term of the gradient.
        cur_skill_diffs = self.match_skill_diffs(skills) + radi_adv
        sigma = self.win_prob(cur_skill_diffs)
        gradient_coef_of_m = np.where(
            self.radiant_win == 1.0,
            (1 - sigma) / self.logistic_scale,
            -sigma / self.logistic_scale
        )
        match_loglik_gradient = \
            (self._sign_of_skill_idx
             * gradient_coef_of_m[self._match_of_skill_idx])

        # Compute Radiant advantage gradient.
        radi_adv_loglik_gradient = np.sum(gradient_coef_of_m)

        # Combine the match likelihoods into a final term.
        match_loglik_gradient = np.append(match_loglik_gradient,
                                          radi_adv_loglik_gradient)

        # Combine all the gradient terms.
        return prior_lprob_gradient + match_loglik_gradient

    def _hessp(self, skills, radi_adv, p, minus_inv_cov_mat):
        """
        Helper function needed by self.fit().

        Compute the matrix multiplication H x p, where H is computed at the
        current coordinates of `skills` and `radi_adv`.

        Argument `minus_inv_cov_mat` is a sparse block diagonal matrix of the
        inverted covariance (sub)matrices including the covariance (matrix) for
        the Radiant advantage term. I.e. the length of `minus_inv_cov_mat` is
        len(skills) + 1.
        """
        cur_skill_diffs = self.match_skill_diffs(skills) + radi_adv
        sigma = self.win_prob(cur_skill_diffs)

        # Compute the prior probability part of the Hessian * p.
        prior_lprob_hessian_p = minus_inv_cov_mat @ p

        # Compute the match likelihood part of the Hessian * p.
        # Hessian coefs are the Hessian coefficients of each match apart
        # from the signs that need to be multiplied in.
        hessian_coefs_of_m = \
            -(sigma * (1 - sigma)) / (self.logistic_scale ** 2)
        signed_p = self._sign_of_skill_idx * p[:-1]
        sign_x_p_x_hessian_coef = \
            signed_p * hessian_coefs_of_m[self._match_of_skill_idx]
        sign_x_p_x_hessian_coef_mat = \
            (sign_x_p_x_hessian_coef[self._skills_by_match_order]
             .reshape(-1, 10))
        hessp_of_match = np.sum(sign_x_p_x_hessian_coef_mat, 1)

        # Add the Radiant advantage term.
        hessp_of_match += hessian_coefs_of_m * p[-1]
        hessp_of_skill_idx = (hessp_of_match[self._match_of_skill_idx]
                              * self._sign_of_skill_idx)

        # Add the last row of the Hessian involving δr δx_i (and multiply
        # by p).
        temp = np.sum(signed_p
                      * hessian_coefs_of_m[self._match_of_skill_idx])
        temp += np.sum(hessian_coefs_of_m) * p[-1]
        hessp_of_skill_idx = np.append(hessp_of_skill_idx, temp)

        total_hessp = prior_lprob_hessian_p + hessp_of_skill_idx
        return total_hessp


class GPVec():
    """
    Helper class used by SkillsGPMCMC: a container for sampling a Gaussian
    process using a standard multivariate normal.

    All the data are internally stored in the standard (multivariate normal)
    space and transformed into the GP space on demand using the covariance
    matrix S.

    The GP covariance function used is exp(-|d|/s), where d is distance
    and s is the scaling factor. Rescaling s through s -> s * u is
    equivalent to exp(-d/(su)) -> exp(-d/s) ** (1/u).

    Args:
        initial_values (numpy.ndarray): Initial values in the untransformed
            (standard) Gaussian space.
        cov_mat (numpy.ndarray): Covariance matrix.
    """
    def __init__(self, initial_values, cov_mat):
        self._validate_args(initial_values, cov_mat)
        self.state = initial_values.astype(np.longdouble)
        self.cov_mat = cov_mat

    def transformed(self, state=None):
        sd_mat = scipy.linalg.cholesky(self.cov_mat, lower=True)
        if state is None:
            state = self.state
        return sd_mat @ state

    def loglik(self, delta=None, cov_func_scale=None):
        """Compute the current (transformed) log-likelihood."""
        state = self.state
        if delta is not None:
            state = np.add(state, delta)
        abs_det = np.linalg.slogdet(self.cov_mat)[1]
        loglik = self._std_normal_logpdf(state) - 0.5 * abs_det
        return loglik

    def delta_loglik(self, delta):
        """
        Difference in log-likelihood based on a delta in the current state.

        The internal state is in standard normal. Therefore, the likelihood
        difference can be computed in a very straightforward manner.
        """
        loglik_diff = -np.sum(delta * (self.state + 0.5 * delta))
        return loglik_diff

    def _validate_args(self, values, cov_mat):
        k = len(values)
        if cov_mat.shape != (k, k):
            raise ValueError("cov_mat is of the wrong shape.")

    def _std_normal_logpdf(self, x):
        res = np.sum(scipy.stats.norm.logpdf(x))
        return res


class GPSample:
    """
    Helper class used by SkillsGPMCMC: a lightweight container for GP
    samples.
    """
    def __init__(self, iter, skills, radi_adv, log_posterior):
        self.iter = iter
        self.skills = skills.astype(np.float64)
        self.radi_adv = radi_adv
        self.log_posterior = log_posterior

    def __str__(self):
        return "[{}] log_posterior: {}, radi_adv: {}, skills: {}".format(
            self.iter, self.log_posterior, self.radi_adv, self.skills)


class SkillsGPMCMC(SkillsGP):
    """Class for sampling from a skills GP model using MCMC.

    Args:
        propose_sd (float): Standard deviation for the proposal Gaussian
            distribution.
        radi_prior_sd (float): Standard deviation for the zero-centred
            prior Gaussian distribution for Radiant advantage. Default: 1.
        radi_offset_proposal_sd (float): Proposal move standard deviation
            for the radiant advantage term. The prior is (currently) assumed
            to be flat.
        save_every_n_iter (int): Save iterations every how many iterations?
            Default: 100.
        initial_sample (GPSample): Initialisation sample. By default, all values
            are initialised at 0.0.

    Attributes:
        samples (pandas.Series): A series of GPSample objects.
        player_skill_vecs (list): A list of GPVec objects of the current
            internal GP states.
    """

    def __init__(self, propose_sd=0.2, radi_offset_proposal_sd=0.1,
                 save_every_n_iter=100, initial_sample=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.propose_sd = propose_sd
        self.radi_offset_proposal_sd = radi_offset_proposal_sd
        self.save_every_n_iter = save_every_n_iter
        self.samples = pd.Series([], index=pd.Int64Index([]), dtype=object)
        self._radi_accept_rate = [0, 0]
        self._skills_accept_rate = [0, 0]

        # Populate the helper Gaussian process vector variables:
        # player_skill_vecs is a self.N long list of GPVec objects for each
        # respective player.
        self.player_skill_vecs = []
        if initial_sample is not None:
            self._cur_iter = initial_sample.iter
            self._cur_radi_adv = initial_sample.radi_adv
            self._cur_log_posterior = initial_sample.log_posterior
            self._cur_skill_diffs = \
                (self.match_skill_diffs(initial_sample.skills)
                 + initial_sample.radi_adv)
            skill_vec_of_player = \
                self.skills_vec_to_skills_by_player(initial_sample.skills)
            inverse_sd_mats = [scipy.linalg.cholesky(x) for x in self.cov_mats]
            for k in range(self.N):
                inverse_sd_mat = inverse_sd_mats[k]
                transformed_skills = skill_vec_of_player[k]
                untransformed_skills = inverse_sd_mat @ transformed_skills
                self.player_skill_vecs.append(
                    GPVec(untransformed_skills, self.cov_mats[k]))
            self.samples[initial_sample.iter] = copy.deepcopy(initial_sample)
        else:
            self._cur_iter = 0
            self._cur_radi_adv = 0.0
            initial_skills = np.repeat(0.0, self.M * 10)
            initial_skills_by_player = \
                self.skills_vec_to_skills_by_player(initial_skills)
            self._cur_skill_diffs = \
                self.match_skill_diffs(initial_skills) + self._cur_radi_adv
            for k in range(self.N):
                initial_values = initial_skills_by_player[k]
                self.player_skill_vecs.append(
                    GPVec(initial_values, self.cov_mats[k]))
            self._cur_log_posterior = \
                self.log_posterior(initial_skills, self._cur_radi_adv)
            self._save_cur_state()

    def radi_win_prob_of_sample(self, sample):
        """Compute the per-match win probability of a sample."""
        skill_diffs = self.match_skill_diffs(sample.skills) + sample.radi_adv
        win_prob = self.win_prob(skill_diffs)
        return win_prob

    def player_skill_by_sample(self, player, sample_slice=slice(None)):
        """Return a matrix (iteration * match) of skills.

        Args:
            player (int): The player ID of interest.
            sample_slice (slice): A slice for slicing iterations.
        """
        player_idx = np.where(self.players_mat.columns == player)[0]
        if len(player_idx) != 1:
            raise ValueError(f"Found {len(player_idx)} matches for {player}.")
        else:
            player_idx = player_idx[0]
        skills_slice = self._skill_slice_of_player[player_idx]
        skill_by_match_mat = np.array([s.skills[skills_slice]
                                       for s in self.samples[sample_slice]])
        iters = pd.Series([s.iter for s in self.samples[sample_slice]],
                          name='iter')
        skills_by_match_df = pd.DataFrame(
            skill_by_match_mat, index=iters,
            columns=self.players_mat.index[self._games_by_player[player_idx]])
        return skills_by_match_df

    def team_skill_by_sample(self, side="radiant", sample_slice=slice(None)):
        """
        Return a matrix of (iteration * match) of total skills of a team
        (either Radiant or Dire).
        """
        team_skill_mat = np.array(
            [self.match_team_skills(sample.skills, side)
             for sample in self.samples[sample_slice]])
        iters = pd.Series([s.iter for s in self.samples[sample_slice]],
                          name='iter')
        team_skill_df = pd.DataFrame(team_skill_mat, index=iters,
                                     columns=self.players_mat.index)
        return team_skill_df

    def radi_adv_by_sample(self, sample_slice=slice(None)):
        """Return a series of Radiant advantages by iteration."""
        samples = self.samples[sample_slice]
        radi_advs = [s.radi_adv for s in samples]
        iters = [s.iter for s in samples]
        return pd.Series(radi_advs, index=pd.Index(iters, name="iter"))

    def radi_win_prob_by_sample(self, sample_slice=slice(None)):
        """Return a matrix (iteration * match) of Radiant win probabilities."""
        win_prob_mat = np.array([self.radi_win_prob_of_sample(sample)
                                 for sample in self.samples[sample_slice]])
        iters = pd.Series([s.iter for s in self.samples[sample_slice]],
                          name="iter")
        win_prob_df = pd.DataFrame(win_prob_mat, index=iters,
                                   columns=self.players_mat.index)
        return win_prob_df

    def iterate(self, n=1):
        """Iterate n times."""

        for i in progressbar.progressbar(range(n)):
            # Update radiant advantage parameter.
            self.iterate_radiant_advantage()

            # Update skill parameters.
            self.iterate_once_player_wise()

            # Increase iteration count. Save current sample?
            self._cur_iter += 1
            if self._cur_iter % self.save_every_n_iter == 0:
                self._save_cur_state()

        # Check the integrity of the running sums.
        cur_skills_vec = self._cur_skills_vec()
        manually_computed_skill_diffs = \
            self.match_skill_diffs(cur_skills_vec) + self._cur_radi_adv
        manually_computed_log_posterior = \
            self.log_posterior(cur_skills_vec, self._cur_radi_adv)
        if not np.allclose(manually_computed_skill_diffs,
                           self._cur_skill_diffs):
            warnings.warn("Loss of integrity with self._cur_skill_diffs")
        if not np.isclose(manually_computed_log_posterior,
                          self._cur_log_posterior):
            warnings.warn("Loss of integrity with self._cur_log_posterior")

    def iterate_radiant_advantage(self):
        """Perform a Metropolis iteration on the Radiant advantage.

        Returns the (potentially) updated value as opposed to modifying `self`
        directly.
        """
        old_radi_prior_lprob = scipy.stats.norm.logpdf(
            self._cur_radi_adv, scale=self.radi_prior_sd)
        old_win_prob = self.win_prob(self._cur_skill_diffs)
        old_match_loglik = np.sum(
            bernoulli_logpmf(self.radiant_win, old_win_prob))

        radi_adv_delta = np.random.normal(scale=self.radi_offset_proposal_sd)
        new_radi_prior_lprob = scipy.stats.norm.logpdf(
            self._cur_radi_adv + radi_adv_delta, scale=self.radi_prior_sd)
        new_win_prob = self.win_prob(self._cur_skill_diffs + radi_adv_delta)
        new_match_loglik = np.sum(
            bernoulli_logpmf(self.radiant_win, new_win_prob))
        log_bayes_factor = (new_radi_prior_lprob - old_radi_prior_lprob
                            + new_match_loglik - old_match_loglik)
        if np.log(np.random.uniform()) < log_bayes_factor:
            self._cur_skill_diffs += radi_adv_delta
            self._cur_radi_adv += radi_adv_delta
            self._cur_log_posterior += log_bayes_factor
            self._radi_accept_rate[0] += 1
        self._radi_accept_rate[1] += 1

    def iterate_once_player_wise(self):
        """Perform a block-wise iteration across all players."""
        for i in range(self.N):
            player_skills_gp = self.player_skill_vecs[i]
            match_idx = self._games_by_player[i]
            skills_delta = np.random.normal(scale=self.propose_sd,
                                            size=len(match_idx))

            # Compute the prior probability portion.
            prior_lprob_change = player_skills_gp.delta_loglik(skills_delta)

            # Compute the match likelihood portion: old match likelihood.
            old_skill_diffs = self._cur_skill_diffs[match_idx]
            radiant_win = self.radiant_win[match_idx]
            old_win_prob = self.win_prob(old_skill_diffs)
            old_loglik = bernoulli_logpmf(radiant_win, old_win_prob)

            # Compute the match likelihood portion: new match likelihood.
            skill_diffs_delta = \
                (self.players_mat.values[match_idx, i]
                 * player_skills_gp.transformed(skills_delta))
            new_skill_diffs = old_skill_diffs + skill_diffs_delta
            new_win_prob = self.win_prob(new_skill_diffs)
            new_loglik = bernoulli_logpmf(radiant_win, new_win_prob)

            # Transition in-place?
            match_loglik_change = new_loglik - old_loglik
            log_bayes_factor = prior_lprob_change + match_loglik_change
            if np.log(np.random.uniform()) < log_bayes_factor:
                player_skills_gp.state += skills_delta
                self._cur_skill_diffs[match_idx] += skill_diffs_delta
                self._cur_log_posterior += log_bayes_factor
                self._skills_accept_rate[0] += 1
            self._skills_accept_rate[1] += 1

    def cur_skills_mat(self):
        """Create the current skills matrix."""
        skills_per_player = [x[0].transformed() for x in self.player_skill_vecs]
        expanded_skills_vecs = \
            [self._expand_sparse_player_vec(skills_per_player[k],
                                            self.player_skill_vecs[k][1])
             for k in range(self.N)]
        skills_mat = np.array(expanded_skills_vecs).T
        return skills_mat

    def _cur_skills_vec(self):
        """Concatenate the skills vectors for each GPVec object."""
        return np.concatenate([x.transformed() for x in self.player_skill_vecs])

    def _save_cur_state(self):
        """Save the current state as a sample."""
        assert self._cur_iter not in self.samples
        skills_by_player = [x.transformed() for x in self.player_skill_vecs]
        sample = GPSample(self._cur_iter, np.concatenate(skills_by_player),
                          self._cur_radi_adv, self._cur_log_posterior)
        self.samples[self._cur_iter] = sample
