"""Fairlay ticker for monitoring odds."""

import logging

import numpy as np
import pandas as pd

from .. import load
from ..models.gp import SkillsGPMAP
from . import fairlay_json_to_df, fetch_markets


def _fetch_fairlay_data():
    """Fetch and filter Fairlay odds data.

    Only keep active non-live single-map markets.
    """
    fairlay_df = fairlay_json_to_df(fetch_markets())
    sel = ((fairlay_df.dota_market_type == 'map')
           & (fairlay_df.Status == 0))  # Only active and non-live matches
    fairlay_df = fairlay_df.loc[sel].reset_index().set_index(
        ['ID', 'Title', 'wager_type', 'norm_runner'])
    return fairlay_df


class FairlayTicker:
    """A Fairlay odds monitoring class.

    Args:
        model_opts (dict): A dictionary of model parameters for
            :class:`models.gp.SkillsGPMAP`. Default: {'cov_func_name':
            'exponential', 'cov_func_kwargs': {'scale': 1.25 * 365 * 24 * 60
            * 60 * 1000}, 'radi_prior_sd': 3.0, 'logistic_scale': 3.0}.
        opts (dict): Options for the ticker's display. Default:
            {'min_train_matches': 30, 'max_days_since_last_game': 60,
            'min_games_with_same_roster': 10}.
    """

    def __init__(self, model_opts=None, opts=None):
        self.model_opts = self._get_model_opts(model_opts)
        self.opts = self._get_opts(opts)
        self.logger = self._setup_logger()
        self.gp_model = None

    def run(self):
        """Fetch and evaluate current Fairlay information."""
        self.matches, self.fairlay_df = self._fetch_data()
        if self.fairlay_df.shape[0] == 0:
            output_df = None
        else:
            self.fairlay_df = self._annotate_fairlay_df()
            self._fit_model()
            self.fairlay_df = self.fairlay_df.merge(
                self._predict_team_1_winprob(),
                left_index=True,
                right_index=True)
            self.fairlay_df = self.fairlay_df.merge(
                self._compute_ev(
                    self.fairlay_df.pred_win_prob_unknown_side),
                left_index=True,
                right_index=True)
            output_df = self._make_output_df()
        return output_df

    def _get_opts(self, user_opts):
        """Return a set of default options."""
        opts = {
            'min_train_matches': 30,
            'max_days_since_last_game': 60,
            'min_games_with_same_roster': 20,
        }
        if user_opts is not None:
            opts.update(user_opts)
        return opts

    def _get_model_opts(self, user_opts):
        """Return a set of default GP model options."""
        model_opts = {
            'cov_func_name': 'exponential',
            'cov_func_kwargs': {'scale': 1.25 * 365 * 24 * 60 * 60 * 1000},
            'radi_prior_sd': 3.0,
            'logistic_scale': 3.0
        }
        if user_opts is not None:
            model_opts.update(user_opts)
        return model_opts

    def _setup_logger(self):
        logger = logging.getLogger("fairlay_ticker_logger")
        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        # Only add the handler if it's not there already.
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger

    def _fetch_data(self):
        self.logger.info('Fetching Datdota matches...')
        # Use both premium and 'professional' matches.
        premium_matches = load.fetch_matches('premium')
        prof_matches = load.fetch_matches('professional')
        matches = load.MatchDF(pd.concat([premium_matches.df, prof_matches.df]))
        self.logger.info('Fetching Fairlay odds...')
        fairlay_df = _fetch_fairlay_data()
        return matches, fairlay_df

    def _fit_model(self):
        self.logger.info('Fitting the model...')
        self.gp_model = SkillsGPMAP.from_match_df(
            self.matches, **self.model_opts)
        self.gp_model.fit()
        self.logger.info('done.')

    def _match_date(self, matches):
        """Compute the match dates of each team by team name.

        Args:
            matches (load.MatchDF): The match data.
        """
        radi_teams = matches.df[['radiant_name', 'radiant_valveId',
                                 'radiant_nicknames', 'radiant_players',
                                 'startDate']]
        dire_teams = matches.df[['dire_name', 'dire_valveId',
                                 'dire_nicknames', 'dire_players',
                                 'startDate']]
        colnames = ['team_name', 'team_id', 'player_names', 'player_ids',
                    'startDate']
        radi_teams.columns = colnames
        dire_teams.columns = colnames
        teams_by_date = (pd.concat([radi_teams, dire_teams])
                         .sort_values('startDate')
                         .reset_index())
        for colname in ('matchId', 'team_id'):
            teams_by_date[colname] = \
                teams_by_date[colname].astype(pd.Int64Dtype())
        teams_by_date['team_name'] = teams_by_date['team_name'].str.lower()
        teams_by_date.index = teams_by_date['team_name']
        return teams_by_date

    def _annotate_fairlay_df(self):
        """Add player IDs and other annotations to self.fairlay_df.

        Columns added are:
        * player IDs of each team, if available.
        * the number of training matches available for each player.
        * the previous time each player played.
        * the number of times the previous roster has stayed the same.

        Returns:
            pandas.DataFrame: a data frame of the above columns.
        """
        # Compute team names and player IDs.
        teams_by_date = self._match_date(self.matches)
        teams_by_last_date = teams_by_date.loc[
            ~teams_by_date.index.duplicated(keep='last')]
        matched_teams_1 = teams_by_last_date.reindex(
                self.fairlay_df['team_1'].str.lower().values)
        matched_teams_1.index = self.fairlay_df.index
        matched_teams_2 = teams_by_last_date.reindex(
            self.fairlay_df['team_2'].str.lower().values)
        matched_teams_2.index = self.fairlay_df.index

        # Compute the number of training matches for each player.
        n_matches_by_player = (self.matches.players_mat != 0).sum()

        def _get_player_n_matches(player_ids):
            if player_ids is np.nan:
                return [np.nan]
            else:
                return n_matches_by_player[list(player_ids)].tolist()
        team_1_n_matches = matched_teams_1.player_ids.apply(
            _get_player_n_matches)
        team_2_n_matches = matched_teams_2.player_ids.apply(
            _get_player_n_matches)

        # Compute the previous time each player played.
        def _get_player_last_match(pids):
            if pids is np.nan:
                return [np.nan]
            else:
                return self.matches.players.last_game.reindex(pids).tolist()
        team_1_last_match = matched_teams_1.player_ids.apply(
            _get_player_last_match)
        team_2_last_match = matched_teams_2.player_ids.apply(
            _get_player_last_match)

        # Compute the (maximum) number of days since the previous game of each
        # team.
        now = pd.Timestamp.now('UTC').tz_convert(None)
        team_1_prev_game_age = \
            (now - team_1_last_match.apply(min).fillna(pd.NaT))
        team_1_prev_game_age = team_1_prev_game_age.dt.days
        team_2_prev_game_age = \
            (now - team_2_last_match.apply(min).fillna(pd.NaT))
        team_2_prev_game_age = team_2_prev_game_age.dt.days

        # Compute the number of times the team has had the same roster.
        teams_by_date['player_ids'] = teams_by_date['player_ids'].apply(
            lambda x: tuple(sorted(x)))
        teams_by_date = (teams_by_date
                         .drop('team_name', 1)
                         .reset_index()
                         .sort_values(['team_name', 'team_id', 'startDate']))
        team_changed = (teams_by_date.team_name
                        != teams_by_date.team_name.shift())
        players_changed = (teams_by_date.player_ids
                           != teams_by_date.player_ids.shift())
        group_idx = (team_changed | players_changed).cumsum()
        roster_n_matches = \
            (teams_by_date
             .groupby([teams_by_date.team_name, teams_by_date.team_id,
                       teams_by_date.player_ids, group_idx], sort=False)
             .size())
        roster_n_matches.index = roster_n_matches.index.get_level_values(0)
        roster_n_matches = (roster_n_matches
                            .reset_index()
                            .drop_duplicates('team_name', keep='last')
                            .set_index('team_name')
                            .squeeze())

        # Use empty string as a placeholder for a team not found.
        roster_n_matches[''] = np.nan
        team_1_roster_n_matches = \
            roster_n_matches[matched_teams_1.team_name.fillna('').str.lower()
                             .values]
        team_2_roster_n_matches = \
            roster_n_matches[matched_teams_2.team_name.fillna('').str.lower()
                             .values]

        return self.fairlay_df.assign(
            team_1_pids=matched_teams_1.player_ids,
            team_1_players=matched_teams_1.player_names,
            team_2_pids=matched_teams_2.player_ids,
            team_2_players=matched_teams_2.player_names,
            team_1_n_matches=team_1_n_matches,
            team_2_n_matches=team_2_n_matches,
            team_1_last_match=team_1_last_match,
            team_2_last_match=team_2_last_match,
            team_1_prev_game_age=team_1_prev_game_age,
            team_2_prev_game_age=team_2_prev_game_age,
            team_1_roster_n_matches=team_1_roster_n_matches.values,
            team_2_roster_n_matches=team_2_roster_n_matches.values
        )

    def _predict_team_1_winprob(self):
        """Predict 'team 1' win probability based on the fitted model."""
        isna = (self.fairlay_df.team_1_pids.isna()
                | self.fairlay_df.team_2_pids.isna())
        if isna.all():
            return pd.Series([np.nan] * self.fairlay_df.shape[0],
                             index=self.fairlay_df.index,
                             name='pred_win_prob_unknown_side')
        else:
            fairlay_pred = self.gp_model.predict_matches(
                self.fairlay_df.team_1_pids.loc[~isna],
                self.fairlay_df.team_2_pids.loc[~isna],
                self.fairlay_df.ClosD[~isna].astype(np.int64) / 1e6)[0]
            fairlay_pred = (fairlay_pred.set_index(self.fairlay_df.index[~isna])
                            .reindex(self.fairlay_df.index))
            return fairlay_pred.pred_win_prob_unknown_side

    def _compute_ev(self, pred_win_prob):
        """Compute EV etc. between self.fairlay_df and pred_win_prob.

        Returns:
            pandas.DataFrame: A data frame with columns for expected value based
                on commission-adjusted odds and break-even odds.
        """
        assert (self.fairlay_df.reset_index().wager_type.isin(['on', 'against'])
                .all())
        ev = np.where(self.fairlay_df.reset_index().wager_type.values == 'on',
                      pred_win_prob * self.fairlay_df.odds_c - 1,
                      (1 - pred_win_prob) * self.fairlay_df.odds_c - 1)
        breakeven_odds = np.where(
            self.fairlay_df.index.get_level_values('wager_type') == 'on',
            1 / pred_win_prob,
            1 / (1 - pred_win_prob))
        out_df = pd.DataFrame(
            {'ev': ev,
             'breakeven_odds': breakeven_odds},
            index=self.fairlay_df.index)
        return out_df

    def _make_output_df(self):
        """Make an output data frame to be printed."""
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

        # Compute warning flag.
        roster_too_new = ((self.fairlay_df.team_1_roster_n_matches
                           < self.opts['min_games_with_same_roster'])
                          | (self.fairlay_df.team_2_roster_n_matches
                             < self.opts['min_games_with_same_roster']))
        too_inactive = ((self.fairlay_df.team_1_prev_game_age
                         > self.opts['max_days_since_last_game'])
                        | (self.fairlay_df.team_2_prev_game_age
                           > self.opts['max_days_since_last_game']))
        too_little_data = ((self.fairlay_df.team_1_n_matches.apply(min)
                            < self.opts['min_train_matches'])
                           | (self.fairlay_df.team_2_n_matches.apply(min)
                              < self.opts['min_train_matches']))
        warning_flag = (
            pd.Series(np.where(np.logical_not(too_little_data), ' ', 'T'))
            + pd.Series(np.where(np.logical_not(roster_too_new), ' ', 'R'))
            + pd.Series(np.where(np.logical_not(too_inactive), ' ', 'D')))

        # Compute EV indicator.
        ev_flag = \
            np.where(np.isnan(self.fairlay_df.ev)
                     | (self.fairlay_df.ev < 0), '     ',
            np.where(self.fairlay_df.ev < 0.05,  '*    ',
            np.where(self.fairlay_df.ev < 0.1,   '**   ',
            np.where(self.fairlay_df.ev < 0.2,   '***  ',
            np.where(self.fairlay_df.ev < 0.3,   '**** ', '*****'))))) # noqa

        int64 = pd.Int64Dtype()
        out_df = pd.DataFrame(
            {'timestamp': timestamp,
             'TRD': warning_flag.values,
             'ev_flag': ev_flag,
             'comp': self.fairlay_df.Comp,
             'descr': self.fairlay_df.Descr,
             'ClosD': self.fairlay_df.ClosD,
             'odds_c': self.fairlay_df.odds_c,
             'p_win': self.fairlay_df.pred_win_prob_unknown_side,
             'ev': self.fairlay_df.ev,
             'breakeven_odds': self.fairlay_df.breakeven_odds,
             't1_train':
                 self.fairlay_df.team_1_n_matches.apply(min).astype(int64),
             't2_train':
                 self.fairlay_df.team_2_n_matches.apply(min).astype(int64),
             't1_roster_xp': self.fairlay_df.team_1_roster_n_matches
                .astype(int64),
             't2_roster_xp': self.fairlay_df.team_2_roster_n_matches
                .astype(int64),
             't1_prev_match': self.fairlay_df.team_1_prev_game_age
                .astype(int64),
             't2_prev_match': self.fairlay_df.team_2_prev_game_age
                .astype(int64),
             't1_players': self.fairlay_df.team_1_players,
             't2_players': self.fairlay_df.team_2_players})
        return out_df


def main():
    ticker = FairlayTicker()
    ticker.logger.setLevel(logging.INFO)
    out_df = ticker.run()
    if out_df is None:
        ticker.logger.info('No Fairlay data.')
    else:
        out_df.reset_index(inplace=True)
        out_df['comp'] = out_df.comp.str.replace(r'^Dota 2 - ', '')
        new_cols = ['timestamp', 'TRD', 'ev_flag', 'comp', 'Title', 'descr',
                    'wager_type', 'norm_runner']
        new_cols = (new_cols
                    + list(out_df.columns[~out_df.columns.isin(new_cols)]))
        with pd.option_context('display.max_colwidth', -1):
            print(out_df[new_cols].to_string(index=False, header=True))


if __name__ == '__main__':
    main()
