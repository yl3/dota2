"""Functions for statistical analysis of the dataset."""


import numpy as np
import pandas as pd
import plotly.graph_objects as plotly_go
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

    def __init__(self, matches, match_pred, skills_mat=None):
        self._validate_match_pred(match_pred)
        self.match_pred = match_pred
        if skills_mat is not None:
            self.skills_mat = skills_mat.reindex(self.match_pred.index)
        else:
            skills_mat = None
        assert isinstance(matches, load.MatchDF)
        self.matches = load.MatchDF(matches.df.drop('series_start_time', 1)
                                    .reindex(self.match_pred.index))

        # Need to re-reindex everything, since load.MatchDF sorts the data
        # frame.
        self.match_pred = self.match_pred.loc[self.matches.df.index]
        self.skills_mat = self.skills_mat.loc[self.matches.df.index]
        
        self._hovertext = None

    @property
    def hovertext(self):
        if self._hovertext is None:
            self._hovertext = self._matches_to_hovertext()
        return self._hovertext

    def player_skills_plot(self, player_ids):
        """Create a Plotly figure showing players' skills."""
        plotly_data = []
        for pid in player_ids:
            match_idx = self.matches.players_mat.loc[:, pid] != 0.0
            player_name = self.matches.players.loc[pid, "name"]
            skills = self.skills_mat.loc[match_idx, pid]
            hovertext = [x + "</br>{}: {:.3f}".format(player_name, s)
                         for x, s in zip(self._hovertext[match_idx], skills)]
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
                         for x, skill in zip(self._hovertext[match_idx],
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
