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
    idx = ['nmatch', 'nwin', 'exp_win', 'poibin_pval', 'pois_pval']
    return pd.Series(res, index=idx)


def _prediction_to_plotly_roc_data(y_true, y_pred, name=None):
    """Return a Plotly data object for plotting a ROC curve."""
    fpr, tpr, thres = sklearn.metrics.roc_curve(y_true, y_pred)
    auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
    if name is None:
        name = "AUC: {:.3f}".format(auc)
    else:
        name = name + " ({:.3f})".format(auc)
    data = plotly_go.Scatter(x=fpr, y=tpr, mode='lines', name=name,
                             showlegend=True, hovertext=np.round(thres, 3))
    return data


def _roc_curve_plotly_plot(data):
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


class MatchPred:
    """Class for storing and analysing match predictions."""

    def __init__(self, matches, match_pred, logistic_scale, skills_mat=None):
        self.logistic_scale = logistic_scale

        self._validate_match_pred(match_pred)
        self.match_pred = self._add_unknown_side_win_prob(match_pred)
        assert isinstance(matches, load.MatchDF)
        self.matches = load.MatchDF(matches.df.reindex(self.match_pred.index))

        # Need to re-reindex everything, since load.MatchDF sorts the data
        # frame by match ID..
        self.match_pred = self.match_pred.loc[self.matches.df.index]
        if skills_mat is not None:
            self.skills_mat = skills_mat.loc[self.matches.df.index]
        else:
            self.skills_mat = None

        self._hovertext = None

    @property
    def hovertext(self):
        if self._hovertext is None:
            self._hovertext = self._matches_to_hovertext()
        return self._hovertext

    def win_prob(self, skill_diffs):
        return gp.logistic_win_prob(skill_diffs, self.logistic_scale)

    def player_skills_plot(self, player_ids):
        """Create a Plotly figure showing players' skills."""
        plotly_data = []
        for pid in player_ids:
            match_idx = self.matches.players_mat.loc[:, pid] != 0.0
            player_name = self.matches.players.loc[pid, "name"]
            skills = self.skills_mat.loc[match_idx, pid]
            hovertext = [x + "</br>{}: {:.3f}".format(player_name, s)
                         for x, s in zip(self.hovertext[match_idx], skills)]
            data = self._match_data_to_graph_obj(
                self.matches.df.loc[:, "startDate"][match_idx].values,
                skills,
                self.matches.players_mat.loc[:, pid][match_idx] == 1.0,
                self.matches.df.loc[:, "radiantVictory"][match_idx],
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
            match_idx = ((self.matches.df.radiant_valveId == tid)
                         | (self.matches.df.dire_valveId == tid))
            is_radi = self.matches.df.radiant_valveId[match_idx] == tid
            team_skill = np.where(is_radi,
                                  self.match_pred.radi_skill[match_idx],
                                  self.match_pred.dire_skill[match_idx])
            team_name = self.matches.teams[tid]
            hovertext = [x + "</br>{}: {:.3f}".format(team_name, skill)
                         for x, skill in zip(self.hovertext[match_idx],
                                             team_skill)]
            data = self._match_data_to_graph_obj(
                self.matches.df.loc[:, "startDate"][match_idx].values,
                team_skill,
                is_radi,
                self.matches.df.loc[:, "radiantVictory"][match_idx],
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
        y_true = self.matches.df.radiantVictory
        if side_known:
            y_pred = self.match_pred.pred_win_prob
        else:
            y_pred = self.match_pred.pred_win_prob_unknown_side
        print(y_pred.head(10))
        if loc_dict is not None:
            for name, loc in loc_dict.items():
                trace = _prediction_to_plotly_roc_data(
                    y_true[loc], y_pred[loc], name)
                traces.append(trace)
        elif iloc_dict is not None:
            for name, iloc in iloc_dict.items():
                trace = _prediction_to_plotly_roc_data(
                    y_true.iloc[iloc], y_pred.iloc[iloc], name)
                traces.append(trace)
        else:
            trace = _prediction_to_plotly_roc_data(y_true, y_pred)
            traces.append(trace)
        return _roc_curve_plotly_plot(traces)

    def match_pred_df_bias(self, win_prob_breaks=np.arange(0.0, 1.1, 0.1),
                           iloc=slice(None), loc=None, side_known=True):
        """Compute prediction bias in a match prediction data frame."""
        if side_known:
            temp = self.match_pred[["pred_win_prob"]].merge(
                self.matches.df[["radiantVictory"]], left_index=True,
                right_index=True)
            if loc is None:
                temp = temp.iloc[iloc]
            else:
                temp = temp.loc[loc]
        else:
            temp = self.match_pred[["pred_win_prob_unknown_side"]].merge(
                self.matches.df[["radiantVictory"]], left_index=True,
                right_index=True)
            temp.rename(columns={'pred_win_prob_unknown_side': 'pred_win_prob'},
                        inplace=True)
            if loc is None:
                temp = temp.iloc[iloc]
            else:
                temp = temp.loc[loc]
        win_prob_interval = \
            pd.cut(temp.pred_win_prob, win_prob_breaks, include_lowest=True)
        grouped = temp.groupby(win_prob_interval)
        res_df = pd.DataFrame(
            [pd.Series(
                win_oe_pval_series(df.pred_win_prob, df.radiantVictory),
                name=name)
             for name, df in grouped])
        res_df.loc[:, "exp_win"] = res_df.loc[:, "exp_win"].round(3)
        return res_df

    def team_skills(self):
        """Most recent skills of each team."""
        radi_skills = pd.DataFrame({'time': self.matches.df.startDate,
                                    'team': self.matches.df.radiant_valveId,
                                    'skill': self.match_pred.radi_skill})
        dire_skills = pd.DataFrame({'time': self.matches.df.startDate,
                                    'team': self.matches.df.dire_valveId,
                                    'skill': self.match_pred.dire_skill})
        combined_skills = (pd.concat([radi_skills, dire_skills])
                           .sort_values('time'))
        combined_skills = combined_skills.loc[
            ~combined_skills.team.duplicated(keep='last')]
        combined_skills = combined_skills.sort_values('skill', ascending=False)
        combined_skills['team_name'] = \
            self.matches.teams[combined_skills.team].values
        combined_skills.set_index('team', inplace=True)
        return combined_skills

    def _validate_match_pred(self, match_pred):
        """Validate columns in the match predictions table."""
        expected_columns = sorted([
            'startTimestamp',
            'pred_win_prob',
            'win_prob-2sd',
            'win_prob+2sd',
            'radi_skill',
            'radi_skill_sd',
            'dire_skill',
            'dire_skill_sd',
            'radi_adv'])
        if not all(match_pred.columns.sort_values() == expected_columns):
            raise ValueError("Unexpected columns in match_pred.")

    def _add_unknown_side_win_prob(self, match_pred):
        """
        Add a column called 'pred_win_prob_unknown_side' for win probability
        given unknown side each team plays on.
        """
        skill_diff = match_pred.radi_skill - match_pred.dire_skill
        radi_adv = match_pred.radi_adv
        y_pred = (self.win_prob(skill_diff + radi_adv) / 2
                  + self.win_prob(skill_diff - radi_adv) / 2)
        return match_pred.assign(pred_win_prob_unknown_side=y_pred)

    def _match_entry_to_hovertext(self, match, prediction, cur_skills):
        """Convert one entry of a match with predictions to hovertext.

        Args:
            match (pandas.Series): A row in self.matches.df.
            prediction (pandas.Series): A row in self.match_pred.
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

    def _matches_to_hovertext(self):
        """Create hovertext for plotly data points."""
        hovertext = [self._match_entry_to_hovertext(m[1], p[1], s[1])
                     for m, p, s in zip(self.matches.df.iterrows(),
                                        self.match_pred.iterrows(),
                                        self.skills_mat.iterrows())]
        return pd.Series(hovertext, index=self.matches.df.index)

    def _side_outcome_to_symbol(self, is_radi, radi_win):
        """Compute Plotly symbol based on side and outcome."""
        symbol = np.where(is_radi,
                          np.where(radi_win, "triangle-up", "triangle-down"),
                          np.where(radi_win, "triangle-down-open",
                                   "triangle-up-open"))
        return symbol

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
