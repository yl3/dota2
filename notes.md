# Notes

## 2019-08-29

Manual inspection of the TI9 group stage data shows that the top predicted EV events are the biggest losers. We are currently predicting roughly even win probabilities too often when games should be more one-sided. Of 20 highest predicted EV maps, the total expected EV is 25 but the observed outcome is -14.

Thus, either the strength of regularisation should be decreased (i.e. lower logistic scale) or the strength of autocorrelation should be decreased (i.e. lower covariance function scale).

## 2019-08-23

TODO:

* Refactor so that predicting of a second `MatchDF` is possible in `MatchDF` itself.
* Refactor to speed up the code using numba.
* Backtest different prior SD and logistic scale parameters.
* Count the number of matches played until a certain match in MatchDF. When backtesting, only consider matches where players have been trained with at least 10-20 matches.

### Checking the difference between Fairlay betting closing times and match start times.

The times are for their TI9 group stage matches

* Secret vs Alliance
    * was scheduled to start August 14, 2019 - 22:00 EDT.
    * The actual start time was recorded as 2019-08-14 22:03:30 EDT.
    * At open, the closing time was set to 2019-08-14 21:00:00 EDT.
    * During the game, the closing time was updated to 2019-08-14 23:00:00 EDT.
    * Finally, the closing date was set to 2019-08-15 02:00:00 EDT.
* Secret vs Newbee
    * Scheduled to start August 15, 2019 - 03:30 EDT.
    * Starting time recorded as 2019-08-15 7:49:08 UTC.
    * Initial closing date was set to 2019-08-15 02:00:00 EDT.
    * Over the game, closing date for map 1 and series spread was revised to 2019-08-15 03:00:00 EDT, then to 4:30 and even 6:30 EDT.
    * Weirdly, the final closing date for map 1 and match spread was 2019-08-16 03:00:00 EDT, whereas the final closing date for map 2 was 2019-08-15 10:37:21 EDT.
* Navi vs Infamous
    * Scheduled to start August 15, 2019 - 21:00 EDT.
    * Recorded as started on at 2019-08-15 21:18:14 EDT.
    * The initial closing date was set to 2019-08-15 21:00:00 EDT.
    * Over time, the closing date for each match/series market was updated to as late as 2019-08-16 02:00:00 EDT.

Thus, the initial close time seems to always be the series start time (even for markets of map 2 onwards). The final closing time is typically 3-4 hours after the start of the series.

## 2019-08-22

**Why does Fnatic's skill decrease after its first win against EG?**

## 2019-08-20

TODO: refactor `src.stats.MatchPred` to contain a `src.stats.MatchPred.df` that has the matches and predictions merged.

## 2019-08-17

### Decimal odds.

Let the *ask* price be $\alpha > 1$. This means that when playing 1 unit, the reward if the selected runner wins is $\alpha$. This is a breakeven bet when $p_{\text{win}} \times (\alpha - 1) = 1 - p_{\text{win}} \iff 1 = p_{\text{win}} \alpha \iff p_{\text{win}} = 1 / \alpha$.

When the *bid* price is $\beta > 1$, it means that for a 1 unit wager on a runner, the accepter of the bid agrees to pay $\beta - 1$ if the selected runner wins. Equivalently, for paying (i.e. losing) 1 units if the runner wins, the accepted bidding amount is $o = 1 / (\beta - 1)$.

Wager amount | Change if runner wins | Change if runner loses
:-----------:|:---------------------:|:---------------------:
$\beta$      | $- (\beta - 1)$       | $+1$
$1 / \beta$  | $- 1$                 | $+1 / (\beta - 1)$

Thus, the equivalent decimal odds for wagering 1 units against the runner is $1 + \frac{1}{\beta - 1} = \frac{\beta - 1 + 1}{\beta - 1} = \frac{\beta}{\beta - 1}$.

## 2019-08-16

**TODO:**

* Hyperparameter optimisation (using BFGS?).
* Munge and analyse Fairlay data.
* Manually annotate the `best_of` and stage (group vs playoff) of each series.

## 2019-08-12

While writing classes to test the first batch of the iterative fitting results, the next things to test through (back)testing are:

1. The optimal MCMC proposal parameters to get close to around 0.234 acceptance rate.
    1. Use the full 5,000 match dataset, iteratively find MAP then test different MCMC parameters. Record acceptance rates and the fitted models.
2. Whether using fewer training matches severely deteriorates AUC, likelihood or precision when performing backtesting.
    1. Test the AUC, likelihood and precision of the predictions for the final 1,000 matches (fitted iteratively using Newton-CG) when trained with an initial 1,000, 2,000, 3,000 or 4,000 matches.

## 2019-08-11

Now that we have some basic match backtesting functionality, we just need some series-level backtesting.

1. Firstly, we need to see if a single skill level variable can really determine victory probability, or whether there are non-linear interactions between teams (for instance such that they beat each other in a circular way). To test this, we need to compute match-level pairwise expected and observed win counts between teams.
2. Secondly, we need to start computing series-level outcome probabilities given match-level win probabilities.

- First match of a series updates the probabilities. Thus, the chance of a 2-0 win becomes more extreme towards the first match's winner.
- We need to sample the posterior to fully capture above.

**TODO next:** match outcome probability bias and AUC.

## 2019-08-10

Spec'ing out the plotting functionality. Use cases:

1. Plotting one or multiple players or one or multiple teams in one plot (time vs skill).
    1. Marker time is circle if Radiant, triangle if Dire.
    2. Marker is filled if a match is won, or empty otherwise.
    3. Teams and players are colored individually.
2. Plotting observed and expected log-likelihood in one plot.
3. All data points must also include line segments for +- 2 sd.
4. On hover, all data points show the respective match information.
    1. Match ID.
    2. Radiant and Dire team names.
    3. Radiant and Dire player names.
    4. Player skill and +- 2 sds.
5. Legend shows the player or team IDs (label of each data series).

## 2019-08-09

TODO:

- Iterative fitting in chunks of series [DONE].
- Return matrices of per-player skills and standard deviations [DONE].
- Profiling of the iterative fitting code (from match 4,500 onwards) [DONE].

Some minor profiling of the `backtest.py` code. When fitting a small number of matches (starting point is 100 fitted matches), about 10% of the time is spent on predicting, 25% on adding matches and 50% on fitting.

    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
       123        74         99.0      1.3      0.0              predicted, mu_mat, var_mat = gp_model.predict_matches(
       124        74      11506.0    155.5      0.0                  match_grp.radiant_players,
       125        74       9384.0    126.8      0.0                  match_grp.dire_players,
       126        74    9255090.0 125068.8      9.7                  match_grp.startTimestamp)
       ...
       142        74        237.0      3.2      0.0              gp_model = gp_model.add_matches(new_players_mat,
       143        74       2139.0     28.9      0.0                                              match_grp.startTimestamp,
       144        74   24315298.0 328585.1     25.4                                              match_grp.radiantVictory)
       145        74   54383483.0 734911.9     56.8              gp_model.fit()

When starting from 4,500 fitted matches, all the time is spent on matches.

    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
       104         1          4.0      4.0      0.0      sys.stderr.write(
       105         1         76.0     76.0      0.0          f"[{_cur_time()}] Fitting the {args.training_matches} matches.\n")
       106         1  186984290.0 186984290.0  20.1      gp_model.fit()
    ...
       142        10         29.0      2.9      0.0              gp_model = gp_model.add_matches(new_players_mat,
       143        10        325.0     32.5      0.0                                              match_grp.startTimestamp,
       144        10   45240502.0 4524050.2     4.9                                              match_grp.radiantVictory)
       145        10  692805870.0 69280587.0   74.5              gp_model.fit()


## 2018-08-08

Next thing to do is to decide what hyperparameter values to use. The hyperparameters in question are:

* `radi_prior_sd` - the standard deviation of the prior Gaussian distribution for Radiant advantage.
* `logistic_scale` - the scaling factor $l$ for the logistic regression formula $1 / (1 + \exp(-d / l))$.
* `scale` - the time scale for the covariance function.

### `scale`

How much should skills be autocorrelated? This can be ultimately estimated from the data.

The formula is $\exp(-|d|/s) = \sigma^2 \leftrightarrow s = -|d|/\log \sigma^2$.

| Distance in years | $\mathbf{E}[\text{autocorrelation}]$ | $\mathbf{E}[\text{covariance}]$ | Imputed `scale` |
|:-------------:|:-------------:| -----:|---|
| 0.25 | 0.9-0.95 | 0.81-0.90 | 1.19-2.37 |
| 0.5  | ~0.8 | ~0.64 | ~1.12 |
| 1    | ~0.7 | ~0.49 | ~1.40 |

**Thus, somewhere around 1-1.5 seems like a good choice for the `scale` parameter.**

### `logistic_scale`

The GP prior standard deviation is assumed to be 1. We want to set `logistic_scale` $s$ such that we get the following expected win rates.

The formula is $1 / (1 + \exp(-d / s)) = p \leftrightarrow s = -d / \log ((1-p)/p)$.

| Team skill | Win rate compared with an average team | Estimated scale |
|:---------|:---|:---|
| 10 (five players each 2 sds above the mean) | 19/20 | 3.40 |
| 10 (five players each 2 sds above the mean) | 24/25 | 3.11 |
| 5 (five players each 1 sd above the mean) | 4/5 | 3.11 |
| 5 (five players each 1 sd above the mean) | 7/8 | 2.40 |

**Thus, a good value is probably around 3.**


### `radi_prior_sd`

A priori, we would probably expect two standard deviations of Radiant advantage to be around $50 \% \pm 7.5 \%$. Thus, two standard deviations should equate to a win probability of $57.5 \%$. $p = 1 / (1 + \exp(-d / l)) \leftrightarrow d = -s * \log ((1-p)/p)$. At this win probability, the difference should be 0.91. **So two standard deviations of Radiant prior SD should be 0.91, i.e. one standard deviation is around 0.5.**

## 2019-08-06

The conditional probability of a multivariate normal is provided in Wikipedia [here](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions).

For reference, the fitting results of all but the *last* (oldest) match in the TI9 qualifiers match dataset is the following.

         fun: -7222.4671618023995
         jac: array([-1.00780705e-05, -2.35187119e-06,  1.31262516e-05, ...,
           -1.39675435e-05,  4.02921851e-06,  1.40821844e-05])
     message: 'Optimization terminated successfully.'
        nfev: 15
        nhev: 2279
         nit: 14
        njev: 28
      status: 0
     success: True
           x: array([-0.2865158 , -0.28647442, -0.28642566, ..., -0.16772377,
           -0.16776032, -0.09798433])

The fitting results for the TI9 qualifiers results but the *first* (newest) match is the following.

         fun: -7215.896591841696
         jac: array([-4.42373891e-06,  2.27665835e-06, -6.43913576e-06, ...,
            2.74491658e-06, -6.37987750e-06,  2.63498046e-05])
     message: 'Optimization terminated successfully.'
        nfev: 21
        nhev: 2561
         nit: 18
        njev: 38
      status: 0
     success: True
           x: array([-0.2846862 , -0.28505657, -0.28517459, ..., -0.16446207,
           -0.16434913, -0.07946344])

After incrementally fitting the first (newest) match, we get the following fitting results. The fitting took **600 ms**.

         fun: -7251.013162612773
         jac: array([ 6.18084507e-06, -8.36016304e-06, -5.12338929e-06, ...,
           -8.23186910e-06, -2.32293850e-06, -5.05975558e-05])
     message: 'Optimization terminated successfully.'
        nfev: 9
        nhev: 1163
         nit: 8
        njev: 16
      status: 0
     success: True
           x: array([-0.28478576, -0.28515596, -0.28527381, ..., -0.16480323,
           -0.16469046, -0.08426584])

## 2019-08-04

After refactoring the Newton-CG fitting code, the fitted results (approximately) match with the old code's results.

New results:

         fun: -7251.013162612669
         jac: array([ 2.71426880e-06, -1.04306676e-06, -5.06777520e-06, ...,
           -1.59853553e-05,  1.18502236e-05, -6.97227832e-06])
     message: 'Optimization terminated successfully.'
        nfev: 18
        nhev: 2136
         nit: 16
        njev: 33
      status: 0
     success: True
           x: array([-0.28622649, -0.28618457, -0.28613565, ..., -0.16674437,
           -0.16678098, -0.08426551])

Old results from `10-test_numerical_optimisation.20190731.ipynb`:

         fun: -7251.013162612677
         jac: array([-7.38146256e-06, -5.53450919e-06, -6.69449200e-06, ...,
            2.25451113e-05,  1.21444848e-06,  5.19662969e-06])
     message: 'Optimization terminated successfully.'
        nfev: 18
        nhev: 2154
         nit: 16
        njev: 33
      status: 0
     success: True
           x: array([-0.28622646, -0.28618455, -0.28613563, ..., -0.16674435,
           -0.16678095, -0.08426547])

## 2019-08-02

Tested multipying a vector with the players matrix (large sparse matrix with 1000 columns and 10 non-zero values per row). As expected, the sparse row format works well with this data.

    641      1513    1036286.0    684.9      0.1              minus_inv_cov_mat2 = scipy.sparse.bsr_matrix(minus_inv_cov_mat)
    642      1513     109381.0     72.3      0.0              prior_lprob_hessian_p = minus_inv_cov_mat2 @ p
    643      1513     297690.0    196.8      0.0              minus_inv_cov_mat2 = scipy.sparse.coo_matrix(minus_inv_cov_mat)
    644      1513     109948.0     72.7      0.0              prior_lprob_hessian_p = minus_inv_cov_mat2 @ p
    645      1513     569503.0    376.4      0.0              minus_inv_cov_mat2 = scipy.sparse.csc_matrix(minus_inv_cov_mat)
    646      1513     100190.0     66.2      0.0              prior_lprob_hessian_p = minus_inv_cov_mat2 @ p
    647      1513     510279.0    337.3      0.0              minus_inv_cov_mat2 = scipy.sparse.csr_matrix(minus_inv_cov_mat)
    648      1513     100316.0     66.3      0.0              prior_lprob_hessian_p = minus_inv_cov_mat2 @ p
    649      1513    2859365.0   1889.9      0.2              minus_inv_cov_mat2 = scipy.sparse.dia_matrix(minus_inv_cov_mat)
    650      1513     166325.0    109.9      0.0              prior_lprob_hessian_p = minus_inv_cov_mat2 @ p
    651      1513 1222742307.0 808157.5     91.9              minus_inv_cov_mat2 = scipy.sparse.dok_matrix(minus_inv_cov_mat)
    652      1513   55435132.0  36639.2      4.2              prior_lprob_hessian_p = minus_inv_cov_mat2 @ p
    653      1513   27084943.0  17901.5      2.0              minus_inv_cov_mat2 = scipy.sparse.lil_matrix(minus_inv_cov_mat)
    654      1513   19529244.0  12907.6      1.5              prior_lprob_hessian_p = minus_inv_cov_mat2 @ p

We are now fitting the full 5,000 match dataset in four minutes!

    $ time kernprof -l src/scripts/fit.py --method newton --scale 1 newton_fit_ti9.dill 
    Starting...
    Iteration.
    Iteration.
    Iteration.
    Iteration.
    Iteration.
    Iteration.
    Iteration.
    Iteration.
    Iteration.
    Iteration.
    Iteration.
    Iteration.
    Iteration.
    Iteration.
    Iteration.
    Iteration.
    Iteration.
    Iteration.
    Optimization terminated successfully.
             Current function value: -124344.356934
             Iterations: 18
             Function evaluations: 19
             Gradient evaluations: 36
             Hessian evaluations: 9200
    Wrote profile results to fit.py.lprof
    
    real	2m21.133s
    user	4m4.251s
    sys	0m25.889s

The following two hard to further optimize lines consume most of the time.

    574     18525      29337.0      1.6      0.1                      scipy.stats.multivariate_normal.logpdf(cur_skills,
    575     18525   32946603.0   1778.5     92.7                                                             cov=cov_mats[k]))  # ~ 33 seconds
    
    649      9200   76918388.0   8360.7     86.1              prior_lprob_hessian_p = minus_inv_cov_mat @ p  # ~ 77 seconds

By avoiding the `scipy.stats.multivariate_normal.logpdf()` call, we are shaving another 40 seconds off.

    ...
    Iteration.
    Optimization terminated successfully.
             Current function value: -124344.356934
             Iterations: 18
             Function evaluations: 19
             Gradient evaluations: 36
             Hessian evaluations: 9200
    Wrote profile results to fit.py.lprof
    
    real	1m55.819s
    user	3m20.333s
    sys	0m18.136s

    573                                                           # skill_prior_lprobs.append(
    574                                                           #     scipy.stats.multivariate_normal.logpdf(cur_skills,
    575                                                           #                                            cov=cov_mats[k]))
    576     18525     211987.0     11.4      4.3                  x = inv_sd_mats[k] @ cur_skills
    577     18525      13341.0      0.7      0.3                  abs_det = abs_log_dets[k]
    578     18525    3183719.0    171.9     64.1                  temp = np.sum(scipy.stats.norm.logpdf(x)) - 0.5 * abs_det
    579     18525      22310.0      1.2      0.4                  skill_prior_lprobs.append(temp)

### Incremental Newton-CG fitting

If only one value needs to be updated, we can impute the skills of the new game (based on already fitted player skills) and refit the entire skills vector from this starting point.

In the full dataset, this incremental fitting of one more match's skills took 56 seconds, about a third of the full fit (3 minutes on the notebook). Whereas the full fit required 15 iterations and 16,031 Hessian evaluations, the incremental fit required only 4 iterations and 5,198 Hessian evaluations.

## 2019-07-31

Rather amazingly, the GP model was correctly fitted numerically using Newton-CG at first try. Fitting the TI9 dataset took one minute(!).

Still refactoring and optimisation to be done. After that, we can proceed to perform backtesting.

Fitting the full model on 5000 matches takes 2 hours.

## 2019-07-29

Despite now saving the transformed values at each iteration, sampling rate is still 100 iterations per 29 seconds (tested once on a notebook). So not too much affected.

### Deriving the Hessian function.

Our parameter vector contains of $n + 1$ values. $n$ of these form a skills vector (that fill a skills matrix column by column). The final value is the current Radiant advantage column.

For the skills vector, the Hessian matrix can be decomposed into two components: the prior probability and the match likelihood.

The prior probability Hessian matrix is just a block diagonal matrix composed of the inverse of covariance matrices; see [Gradient and Hessian of the posterior probability](#Gradient-and-Hessian-of-the-posterior-probability.). The final value of this matrix, $H_{n+1, n+1}^\text{prior}$, is simply $\frac{-1}{\text{self.radi_prior_sd}^2}$.

The match log-likelihood part of the Hessian, $H^\text{match}$, is a bit more complicated. This is also a $(n+1) \times (n+1)$ matrix. Let $\sigma(d) = \frac{1}{1 + \exp(-d / l)}$, where $d$ is the skill difference and $l$ is the logistic scaling factor. The twice differentiated match log-likelihoods with respect to parameters $i$ and $j$ is the following.

$$\frac{\partial^2 \text{match logliks}}{\partial x_i \partial x_j} = \sum_{\text{match } m} -\frac{s_{i, m} s_{j, m}}{l^2} (\sigma(d)(1 - \sigma(d))),$$

where $s_{i, m}$ is 1 or 0 depending on whether parameter $i$ is Radiant or Dire in match $m$.

It's noteworthy that each skill participates only in one match, and each match is affected by ten skills and the Radiant advantage parameter.

Thus, in each row $i$ of $H^\text{match}$ but the last one, all but 11 values are zero: the ten values corresponding to the ten skills participating in the match where skill $i$ is, plus the radiant advantage term. The sign of these non-zero values must be multiplied with dot product with the input vector $p$. Thus, for each row $i$, we need two arrays.

* An index array $index_i$ indicating all the players in the current match.
* A sign array, $sign_i$ indicating the side of each respective player.

We also need an overall length $n$ sign vector for each row $i$. Then, $(H^\text{match} p)_i = -\frac{(\sigma(d)(1 - \sigma(d)))}{l^2}\sum sign_i \times p[index_i]$.

## 2019-07-26

By calculating log-prior probability change based on a state change without explicitly computing the log determinant term (which cancels out), we now get to <1 seconds per iteration.

After additional optimisation, which mostly involves writing out scipy.stats functions and abandoning their checks, the time spent on 100 iterations is:

scipy version | time used (seconds) | mean
--------------|---------------------|----- 
`1.2.1-py37_blas_openblash486cb9f_0` | 38, 33, 43, 43, 37 | 38.8
`1.3.0-py37hab3da7d_0` | 37, 38, 38, 33, 34 | 36

We are getting just shy of 3 iterations per second.

A notebook run yielded 0.235 seconds per iteration. Compared with the non-optimised rate of 4.8 seconds per iteration, this is a 20-fold improvement.

The model likelihood saturates after around 5000 iterations.

![](static/10000_matches_per_player_sampling.20190726.png)

On an Amazon instance c5.xlarge, running 100 iterations took precisely 17 seconds on five separate runs.

## 2019-07-25

Added sampling for the radiant advantage.

After refactoring and "optimisation", the speed per iteration is ~1 second. Based on line profiling, all the time is spent on sampling or taking the likelihood of standard multivariate normal distributions.

## 2019-07-23

One more note to yesterdays computations. For the exponential Gaussian process, the covariance function is $cov(x_i, x_j) = \exp(-\frac{|x_i - x_j|}{su}) = \exp(-\frac{|x_i - x_j|}{s})^{1/u}$. Thus, 

## 2019-07-22

Overnight run of the full match dataset (= 5000 matches up to TI9 qualifiers). Rate: 5515 iterations / 7h 21min 23sec = 4.8 sec / iteration.

Code should be optimised by performing a minimum amount matrix multiplications. This is done by random walking the per-sample skills vectors using a standard multivariate normal. The only time the actual skills are needed is when we need to compute the match probabilities.

If $X$ is a $k$ length multivariate normal distribution (column) vector with a covariance matrix of $\Sigma = AA^\top$, then $X \sim AZ$, where $Z \sim \mathcal{N}(\vec{0}, I)$.

We can keep random walking in this "standard normal" space. When needed to compute match likelihoods, a one-off transformation can be performed using a *precomputed and prestored* matrix $A$.

The log density of a multivariate normal distribution is the following.

$$\frac{\exp(-\frac{1}{2}X^\top \Sigma^{-1} X)}{\sqrt{(2\pi)^k |\Sigma|}}$$

Thus, the log-likelihood of $X$ is:

$$-\frac{1}{2} X^\top (AA^\top)^{-1} X - \frac{1}{2} \log((2\pi)^k |\Sigma|)$$

$$= -\frac{1}{2} Z^\top A^\top (A^\top)^{-1} A^{-1} AZ - \frac{1}{2} (\log((2\pi)^k + \log |\Sigma|)$$

$$= -\frac{1}{2} Z^\top I Z - \frac{1}{2} \log(2\pi)^k - \frac{1}{2} \log |\Sigma| $$

$$= P(Z) - \frac{1}{2} \log |\Sigma|$$

Therefore, we can just take the standard normal log-likelihood and subtract precomputed $\log |\Sigma|$.

Secondly, an array should be kept with the running sums of the match skill differences (including Radiant advantage). This way, match differences can be updated simply by updating the changed player values, without having to sum over the unchanged skills.

## 2019-07-21

Next steps:

* Have a high scaling factor for the logistic model to allow more disparity between teams. Essentially, reduce the effect of regularisation.
* Optimise speed.
* Simplify code - by default represent skills column-by-column as opposed to row by row.

Also, it seems like drawing random multivariate normal samples is slower than drawing a random standard multivariate normal, then multiplying it with a precomputed Cholesky decomposed matrix.

## 2019-07-19

Previous attempt on MCMC sampling of the model was marred by a poor proposal distribution that proposed samples that were highly implausible for the covaried normal distributions.

Here is a proposal distribution $q()$ that should work better. We know that if a multivariate normal $X \sim \mathcal{N}(0, \Sigma)$, then $X = AZ$, where $Z \sim \mathcal{N}(0, I)$ and $\Sigma = AA^T$. Equivalently then, $A^{-1}X = Z$. Since $A^{-1}X$ has no covariance, a standard normal proposal distribution should cause no problem. This is equivalent drawing a sample from $Z = z$ and proposing a move of $X \rightarrow X + Az$. Due to the transformation by $A$, the proposed moves take into account the covariance pattern of X, resulting in proposals that are "consistent" with the correlation pattern of X. Yet, the proposal distribution is fully symmetrix, since $q(X+Az \mid X) = \Pr(Z = z) = \Pr(Z = -z) = q(X \mid X + Az)$.

Or even better, we could just draw the next move from $\mathcal{N}(0, \Sigma)$.

## 2019-07-17

In commit `bff2c81f43eaafe8747fcbfeaed5d64d2b97bef4`, the Gaussian process works and is able to perform Metropolis-Hastings sampling on a small dataset with increasing likelihood. See [this notebook](notebooks/02-test_gp_hyperparams.20190718.ipynb).

However, due to the Gaussian process constraints and the information-poor nature of the per-game outcomes, the sampling is completely dominated by the Gaussian prior and not at all by the match outcomes.

An improved approach might be to first find the maximum a posteriori (MAP) of the model, the sample around it using Metropolis-Hastings.

An initial attempt to brute force the fit using BFGS failed. On a small TI9 qualifers dataset, BFGS fitting did not complete after more than 10 hours.

Separately, line profiling was performed on [a script](src/scripts/fit_ti9.py) that fitted the GP on the same TI9 dataset. The results show that computing the multivariate Gaussian log-probability density function consumes the majority of the time during fitting. Since this function is implemented in `scipy.stats.multivariate_normal`, it is unclear how this function could be sped up.

An alternative fitting method might be Newton-conjugate descent (Newton-CG), with the following reasoning. The logarithm of the Bayes formula's numerator for a vector of a player's skills at each match, $\vec{x}$, is

$$\log P(\vec{x} \mid \vec{\mu}, \Sigma) + \sum_{m \text{ in matches}} \log P(\text{outcome}_m \mid \text{skill difference}_m)$$

### Gradient and Hessian of the posterior probability.

The gradient can be computed separately for each term. For the prior probability part, the gradient is simply $\Sigma^{-1} \vec{x}$ ([source](https://stats.stackexchange.com/questions/90134/gradient-of-multivariate-gaussian-log-likelihood)). In the code, we can compute this efficiently using

```python
block_diag_mat = scipy.linalg.block_diag(
    [numpy.linalg.inv(M) for M in self.cov_mat])
return block_diag_mat @ self.skills.T
```

The Hessian of the multivariate normal (w.r.t. means) is simply $-\Sigma^{-1}$ ([source](https://stats.stackexchange.com/questions/27436/how-to-take-derivative-of-multivariate-normal-density)).

In terms of the likelihood with regards to match outcomes, the Jacobian and the Hessian for logistic regression can be found [here](https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function).

With the gradient and the Hessian, the Newton-Raphson method can be used for finding the MAP. 
