"""Functions for statistical analysis of the dataset."""


import numpy as np
import pandas as pd
import plotly.graph_objects as plotly_go
from poibin import PoiBin
import scipy.stats
import scipy.spatial.distance
import sklearn.metrics

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
            pd.DataFrame(team_skills),
            metric=lambda x, y: gp.logistic_win_prob(x - y, logistic_scale))))
    lower = np.tril(scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(
            pd.DataFrame(team_skills),
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
    idx = ['n', 'obs', 'exp', 'poibin_pval', 'pois_pval']
    return pd.Series(res, index=idx)


def _prediction_to_plotly_roc_data(y_true, y_pred, name=None):
    """Return a Plotly data object for plotting a ROC curve."""
    fpr, tpr, thres = sklearn.metrics.roc_curve(y_true, y_pred)
    data = plotly_go.Scatter(x=fpr, y=tpr, mode='lines', name=name,
                             showlegend=True, hovertext=np.round(thres, 3))
    return data


def roc_curve_plotly_plot(data):
    """Create a Plotly ROC curve plot.

    Args:
        data (list): List of `plotly.graph_objs.Scatter` objects for
            plotly.Figure.
    """
    layout = plotly_go.Layout(
        title='ROC curves',
        xaxis=dict(title='Cumulative proportion of all Dire wins'),
        yaxis=dict(title='Cumulative proportion of all Radiant wins'))
    # Add a 0, 1 line.
    null_trace = plotly_go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                   line=dict(color='grey', dash='dash'),
                                   showlegend=False)
    return plotly_go.Figure(data=data + [null_trace], layout=layout)


def _win_prob_bias(pred_win_prob, outcome, win_prob_breaks):
    """
    Generate a data frame of observed and expected wins grouped by
    predicted win probability bins.
    """
    win_prob_interval = \
        pd.cut(pred_win_prob, win_prob_breaks, include_lowest=True)
    temp = pd.DataFrame({'pred_win_prob': pred_win_prob, 'outcome': outcome})
    grouped = temp.groupby(win_prob_interval)
    res_df = pd.DataFrame(
        [pd.Series(
            win_oe_pval_series(df.pred_win_prob, df.outcome), name=name)
         for name, df in grouped])
    res_df.loc[:, "exp"] = res_df.loc[:, "exp"].round(3)
    return res_df


class DotaSeries:
    """A class for a set (aka series) of matches and its win probability.

    Args:
        match_dicts (list): A list of dictionaries with keys for all of
            ['radiant_valveId', 'dire_valveId', 'radiantVictory', 'radi_adv',
            'radiant_name', 'dire_name'].
            The list is expected to be ordered by 'startDate'.
        team_1_match_win_prob (list): The predicted win probability of each
            match in the series for team 1, which is the Radiant team of the
            first match.
    """
    def __init__(self, match_dicts, team_1_match_win_prob):
        self.match_dicts = match_dicts
        self.team_1_match_win_prob = team_1_match_win_prob
        self._set_attrs()

    def from_df(df):
        """
        The data frame conforms to `MatchDF` checks plus must have a
        column called 'pred_win_prob'.
        """
        return DotaSeries(df.to_dict('records'), df.pred_win_prob.iloc[0])

    def to_dict(self):
        """Create a dictionary out of all the useful attributes.

        Returns:
            dict: A dictionary of the following values. 'winner': team ID of the
                winning team or None.
        """
        attrs = ['startDate', 'best_of', 'team_1', 'team_1_name', 'team_2',
                 'team_2_name', 'team_1_score', 'team_2_score', 'winner',
                 'team_1_series_win_prob', 'team_2_series_win_prob',
                 'series_draw_prob', 'team_1_match_win_prob']
        return {k: self.__getattribute__(k) for k in attrs}

    def _set_attrs(self):
        """Compute and set attributes from self.match_dicts."""
        self.team_1 = self.match_dicts[0]['radiant_valveId']
        self.team_1_name = self.match_dicts[0]['radiant_name']
        self.team_2 = self.match_dicts[0]['dire_valveId']
        self.team_2_name = self.match_dicts[0]['dire_name']
        self.startDate = self.match_dicts[0]['startDate']

        # Compute team 1 and team 2's skills by match.
        for match in self.match_dicts:
            if match['radiant_valveId'] == self.team_1:
                match['team_1_win'] = match['radiantVictory']
            else:
                match['team_1_win'] = not match['radiantVictory']

        # Figure out team 1 and team 2 scores and the series length.
        self.team_1_score = self.team_2_score = 0
        self.team_1_score = sum([x['team_1_win']
                                 for x in self.match_dicts])
        self.team_2_score = sum([not x['team_1_win']
                                 for x in self.match_dicts])
        assert self.team_1_score + self.team_2_score == len(self.match_dicts)
        self.series_draw_prob = 0.0
        if self.team_1_score == self.team_2_score:
            self.best_of = 2 * self.team_1_score
            self.winner = None
            self.series_draw_prob = \
                scipy.stats.binom.pmf(self.team_1_score, self.best_of,
                                      self.team_1_match_win_prob)
        elif self.team_1_score > self.team_2_score:
            self.best_of = 2 * self.team_1_score - 1
            self.winner = self.team_1
        elif self.team_1_score < self.team_2_score:
            self.best_of = 2 * self.team_2_score - 1
            self.winner = self.team_2
        else:
            raise Exception("Unexpected scores with matches: \n"
                            + str(self.match_dicts))

        # Finally, Pre-compute the probability of each series outcome
        # combination. In series_outcomes, 1 and 2 correspond to team 1 and 2
        # winning.
        points_to_win = self.best_of // 2 + 1
        self.team_1_series_win_prob = \
            1 - scipy.stats.binom.cdf(points_to_win - 1, self.best_of,
                                      self.team_1_match_win_prob)
        self.team_2_series_win_prob = \
            1 - self.team_1_series_win_prob - self.series_draw_prob


class MatchPred:
    """Class for storing and analysing match predictions.

    Arguments:
        matches (MatchDF): A data frame of matches that were trained and/or
            predicted on.
        match_pred (pandas.DataFrame): A <matches> x <players> data frame of
            skill predictions.

    """

    def __init__(self, matches, match_pred, logistic_scale, skills_mat=None):
        # Store the match and predictions data.
        self._validate_match_pred(match_pred)
        assert isinstance(matches, load.MatchDF)
        self.matches = load.MatchDF(matches.df.reindex(match_pred.index))
        self.df = self.matches.df.merge(match_pred, left_index=True,
                                        right_index=True)

        # Store some other miscellaneous data.
        assert isinstance(logistic_scale, (float, int))
        self.logistic_scale = logistic_scale

        # Store skills matrix if available.
        if skills_mat is not None:
            self.skills_mat = skills_mat.loc[self.df.index]
        else:
            self.skills_mat = None

        self._hovertext = None

    @property
    def match_pred(self):
        """For backward compatibility."""
        return self.df

    @property
    def hovertext(self):
        if self._hovertext is None:
            self._hovertext = self._matches_to_hovertext()
        return self._hovertext

    @property
    def series_df(self):
        """Compute a series data frame."""
        if not hasattr(self, '_series_df', ):
            self._series_df = self._compute_series_df()
        return self._series_df

    def win_prob(self, skill_diffs):
        return gp.logistic_win_prob(skill_diffs, self.logistic_scale)

    def player_skills_plot(self, player_ids):
        """Create a Plotly figure showing players' skills."""
        plotly_data = []
        for pid in player_ids:
            if pid in self.matches.players_mat.columns:
                match_idx = self.matches.players_mat.loc[:, pid] != 0.0
            else:
                match_idx = np.repeat(False, self.matches.players_mat.shape[0])
            player_name = self.matches.players.loc[pid, "name"]
            skills = self.skills_mat.loc[match_idx, pid]
            hovertext = [x + "</br>{}: {:.3f}".format(player_name, s)
                         for x, s in zip(self.hovertext[match_idx], skills)]
            data = self._match_data_to_graph_obj(
                self.df.loc[match_idx, "startDate"].values,
                skills,
                self.players_mat.loc[match_idx, pid] == 1.0,
                self.df.loc[match_idx, "radiantVictory"],
                hovertext,
                name=player_name)
            plotly_data.append(data)
        layout = plotly_go.Layout(
            yaxis=plotly_go.layout.YAxis(title="Player skill"))
        fig = plotly_go.Figure(data=plotly_data, layout=layout)
        return fig

    def team_skills_plot(self, team_ids):
        """Create a Plotly figure showing teams' skills."""
        plotly_data = []
        for tid in team_ids:
            match_idx = ((self.df.radiant_valveId == tid)
                         | (self.df.dire_valveId == tid))
            is_radi = self.df.radiant_valveId[match_idx] == tid
            team_skill = np.where(is_radi,
                                  self.df.radi_skill[match_idx],
                                  self.df.dire_skill[match_idx])
            team_name = self.matches.teams[tid]
            hovertext = [x + "</br>{}: {:.3f}".format(team_name, skill)
                         for x, skill in zip(self.hovertext[match_idx],
                                             team_skill)]
            data = self._match_data_to_graph_obj(
                self.df.loc[match_idx, "startDate"].values,
                team_skill,
                is_radi,
                self.df.loc[match_idx, "radiantVictory"],
                hovertext,
                name=team_name)
            plotly_data.append(data)
        layout = plotly_go.Layout(
            yaxis=plotly_go.layout.YAxis(title="Team skill"))
        fig = plotly_go.Figure(data=plotly_data, layout=layout)
        return fig

    def roc_curve_plot(self, side_known=True, loc_dict=None, iloc_dict=None):
        """Plot a ROC curve of the current predictions.

        Args:
            side_known (bool): Assume that the side of each team is known?
            loc_dict (dict): A dictionary of slices for slicing matches.
            iloc_dict (dict): A dictionary of iloc slices for slicing matches.
        """
        traces = []
        y_true = self.df.radiantVictory
        if side_known:
            y_pred = self.df.pred_win_prob
        else:
            y_pred = self.df.pred_win_prob_unknown_side
        if loc_dict is not None:
            for name, loc in loc_dict.items():
                name += f", AUC: {self._roc_auc(side_known, loc=loc)}"
                trace = _prediction_to_plotly_roc_data(
                    y_true[loc], y_pred[loc], name)
                traces.append(trace)
        elif iloc_dict is not None:
            for name, iloc in iloc_dict.items():
                name += f", AUC: {self._roc_auc(side_known, iloc=iloc)}"
                trace = _prediction_to_plotly_roc_data(
                    y_true.iloc[iloc], y_pred.iloc[iloc], name)
                traces.append(trace)
        else:
            name = f"AUC: {self._roc_auc(side_known)}"
            trace = _prediction_to_plotly_roc_data(y_true, y_pred, name)
            traces.append(trace)
        return roc_curve_plotly_plot(traces)

    def match_pred_df_bias(self, win_prob_breaks=np.linspace(0.0, 1.0, 11),
                           iloc=slice(None), loc=None, assume_side_known=True):
        """Compute prediction bias in a match prediction data frame."""
        if assume_side_known:
            temp = self.df[["pred_win_prob"]].merge(
                self.df[["radiantVictory"]], left_index=True, right_index=True)
            if loc is None:
                temp = temp.iloc[iloc]
            else:
                temp = temp.loc[loc]
        else:
            temp = self.df[["pred_win_prob_unknown_side"]].merge(
                self.df[["radiantVictory"]], left_index=True, right_index=True)
            temp.rename(columns={'pred_win_prob_unknown_side': 'pred_win_prob'},
                        inplace=True)
            if loc is None:
                temp = temp.iloc[iloc]
            else:
                temp = temp.loc[loc]
        res_df = _win_prob_bias(temp.pred_win_prob, temp.radiantVictory,
                                win_prob_breaks)
        return res_df

    def series_pred_df_bias(self, win_prob_breaks=np.arange(0.0, 1.1, 0.1),
                            team_ids=None, iloc=None, loc=None, draw=False):
        """Compute prediction bias for team 1 series victory.

        Args:
            win_prob_breaks (array-like): An array of prediction probability
                breaks for the bins.
            teams (list): List of teams (team IDs) to return results on.
            iloc (slice): Subset self.series_df with this slice before computing
                bias.
            loc (slice): Subset self.series_df with this slice before computing
                bias. Ignored if iloc is not None.
            draw (bool): Compute the draw probability instead of team 1 win
                probability?
        """
        if draw:
            raise NotImplementedError(
                "draw=True is not supported yet, since whether a 2-0 match is "
                "a bo2 or bo3 cannot be inferred.")
        if iloc is not None:
            series_df = self.series_df.iloc[iloc]
        elif loc is not None:
            series_df = self.series_df.loc[loc]
        else:
            series_df = self.series_df
        is_best_of_even = (series_df.best_of // 2) * 2 == series_df.best_of
        if draw:
            series_df = series_df.loc[is_best_of_even]
        else:
            series_df = series_df.loc[~is_best_of_even]
        if team_ids is None:
            if not draw:
                output = _win_prob_bias(series_df.team_1_series_win_prob,
                                        series_df.winner == series_df.team_1,
                                        win_prob_breaks)
            else:
                output = _win_prob_bias(series_df.series_draw_prob,
                                        series_df.winner.isna(),
                                        win_prob_breaks)
        else:
            output = []
            for team_id in team_ids:
                idx = ((series_df.team_1 == team_id)
                       | (series_df.team_2 == team_id))
                series_df_f = series_df.loc[idx]
                if not draw:
                    win_prob = np.where(
                        series_df_f.team_1 == team_id,
                        series_df_f.team_1_series_win_prob,
                        series_df_f.team_2_series_win_prob)
                    team_wins = np.where(
                        series_df_f.team_1 == team_id,
                        series_df_f.winner == series_df_f.team_1,
                        series_df_f.winner == series_df_f.team_2)
                else:
                    win_prob = series_df_f.series_draw_prob
                    team_wins = series_df_f.winner.isna()
                win_prob_bias_df = _win_prob_bias(win_prob, team_wins,
                                                  win_prob_breaks)
                output.append((team_id, win_prob_bias_df))
        return output

    def match_loglik(self, assume_side_known=True, team_ids=None):
        if assume_side_known:
            pred_win_prob = self.df.pred_win_prob
        else:
            pred_win_prob = self.df.pred_win_prob_unknown_side
        loglik = np.log(np.where(self.df.radiantVictory, pred_win_prob,
                                 1 - pred_win_prob))
        if team_ids is not None:
            idx = self.matches.loc_team(team_ids)
            loglik = loglik[idx]
        return loglik

    def team_skills(self):
        """Most recent skills of each team."""
        radi_skills = pd.DataFrame({'time': self.df.startDate,
                                    'team': self.df.radiant_valveId,
                                    'skill': self.df.radi_skill})
        dire_skills = pd.DataFrame({'time': self.df.startDate,
                                    'team': self.df.dire_valveId,
                                    'skill': self.df.dire_skill})
        combined_skills = (pd.concat([radi_skills, dire_skills])
                           .sort_values('time'))
        combined_skills = combined_skills.loc[
            ~combined_skills.team.duplicated(keep='last')]
        combined_skills = combined_skills.sort_values('skill', ascending=False)
        combined_skills['team_name'] = \
            self.matches.teams[combined_skills.team].values
        combined_skills.set_index('team', inplace=True)
        return combined_skills

    def pairwise_win_oe_mat(self, team_ids, assume_side_known=True):
        """Compute the observed and expected win counts among a group of teams.

        Args:
            team_ids (iterable): A list of team IDs to consider.
            assume_side_known (bool): Whether to assume that the side of each
                team was known beforehand.

        Returns:
            pandas.DataFrame: A data frame of total matchups between the teams.
            pandas.DataFrame: A data frame of expected wins between the teams.
            pandas.DataFrame: A data frame of observed wins between the teams.
        """
        idx = self.matches.loc_team(team_ids, and_operator=True)
        matches_f = self.df.loc[idx]
        total = (matches_f.groupby(['radiant_name', 'dire_name']).radiantVictory
                 .size().unstack())
        wins = (matches_f.groupby(['radiant_name', 'dire_name']).radiantVictory
                .sum().unstack())
        if assume_side_known:
            merged_df = matches_f.merge(self.df[['pred_win_prob']],
                                        left_index=True, right_index=True)
            expected = (merged_df.groupby(['radiant_name', 'dire_name'])
                        .pred_win_prob.sum().unstack())
        else:
            merged_df = matches_f.merge(
                self.df[['pred_win_prob_unknown_side']],
                left_index=True,
                right_index=True)
            expected = (merged_df.groupby(['radiant_name', 'dire_name'])
                        .pred_win_prob_unknown_side.sum().unstack())
        return total, expected, wins

    def _validate_match_pred(self, match_pred):
        """Validate columns in the match predictions table."""
        expected_columns = set([
            'startTimestamp',
            'pred_win_prob',
            'win_prob-2sd',
            'win_prob+2sd',
            'radi_skill',
            'radi_skill_sd',
            'dire_skill',
            'dire_skill_sd',
            'radi_adv',
            'pred_win_prob_unknown_side'])
        if not set(match_pred.columns) >= expected_columns:
            raise ValueError("Missing expected columns in match_pred.")

    def _match_entry_to_hovertext(self, match, prediction, cur_skills):
        """Convert one entry of a match with predictions to hovertext.

        Args:
            match (pandas.Series): A row in self.df.
            prediction (pandas.Series): A row in self.df.
            cur_skills (pandas.Series): A row in self.skills_mat.
        """
        radi_skills = cur_skills.loc[match.radiant_players]
        dire_skills = cur_skills.loc[match.dire_players]
        radi_roster = "</br>".join(
            ["{} ({}) | <b>{:.2f}</b>".format(name, id, skill)
             for name, id, skill in zip(match.radiant_nicknames,
                                        match.radiant_players, radi_skills)])
        dire_roster = "</br>".join(
            ["{} ({}) | <b>{:.2f}</b>".format(name, id, skill)
             for name, id, skill in zip(match.dire_nicknames,
                                        match.dire_players, dire_skills)])
        title = "<b>{} | {}</b>".format(
            match.startDate.strftime("%Y-%m-%d %H:%M"), match.name)
        outcome = "1 - 0" if match.radiantVictory else "0 - 1"
        subtitle = "{} | predicted: {:.3f}".format(outcome,
                                                   prediction.pred_win_prob)
        radi_title = "<b>{} ({}) | {:.2f}</b>".format(
            match.radiant_name, match.radiant_valveId, prediction.radi_skill)
        dire_title = "<b>{} ({}) | {:.2f}</b>".format(
            match.dire_name, match.dire_valveId, prediction.dire_skill)
        output = "</br>".join([
            title, "",
            subtitle, "",
            radi_title,
            radi_roster, "",
            dire_title,
            dire_roster, "",
            "radi_adv: <b>{:.2f}</b>".format(prediction.radi_adv)
        ])
        return(output)

    def _compute_series_df(self):
        skill_diff = self.df.radi_skill - self.df.dire_skill
        radi_adv = self.df.radi_adv
        y_pred = (self.win_prob(skill_diff + radi_adv) / 2
                  + self.win_prob(skill_diff - radi_adv) / 2)
        series_df = \
            (self.df.assign(pred_win_prob=y_pred)
             .groupby('seriesId')
             .apply(lambda df: pd.Series(DotaSeries.from_df(df).to_dict())))
        return series_df

    def _matches_to_hovertext(self):
        """Create hovertext for plotly data points."""
        hovertext = [self._match_entry_to_hovertext(m[1], p[1], s[1])
                     for m, p, s in zip(self.df.iterrows(),
                                        self.df.iterrows(),
                                        self.skills_mat.iterrows())]
        return pd.Series(hovertext, index=self.df.index)

    def _side_outcome_to_symbol(self, is_radi, radi_win):
        """Compute Plotly symbol based on side and outcome."""
        symbol = np.where(is_radi,
                          np.where(radi_win, "triangle-up", "triangle-down"),
                          np.where(radi_win, "triangle-down-open",
                                   "triangle-up-open"))
        return symbol

    def _roc_auc(self, side_known, loc=None, iloc=None):
        y_true = self.df.radiantVictory
        if side_known:
            y_pred = self.df.pred_win_prob
        else:
            y_pred = self.df.pred_win_prob_unknown_side
        if loc is not None:
            y_true, y_pred = (y_true[loc], y_pred[loc])
        elif iloc is not None:
            y_true, y_pred = (y_true.iloc[iloc], y_pred.iloc[iloc])
        auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
        return auc

    def _match_data_to_graph_obj(self, time, y, is_radi, radi_win, hovertext,
                                 name=None):
        """Convert data into a Plotly graph object.

        Args:
            time (array-like): The time of a match.
            y (array-like): Y-axis value to be plotted.
            is_radi (array-like): Boolean array of whether a player/team is on
                Radiant.
            radi_win (array-like): Boolean array of whether Radiant won.
            hovertext (array-like): A list of hovertexts for each data point.
            name (str): Name of the series.
        """
        df = pd.DataFrame({'x': time, 'y': y, 'is_radi': is_radi,
                           'radi_win': radi_win})
        df = df.sort_values('x')
        marker = {
            'symbol': self._side_outcome_to_symbol(df.is_radi, df.radi_win),
            'size': 8, 'opacity': 0.7}
        line = {'width': 1}
        data = plotly_go.Scatter(x=df.x, y=df.y, marker=marker,
                                 mode='lines+markers', line=line, name=name,
                                 showlegend=True, hovertext=hovertext,
                                 hoverinfo="text")
        return data
