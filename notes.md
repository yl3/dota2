# Notes

## 2019-07-21

Next steps:

* Have a high scaling factor for the logistic model to allow more disparity between teams. Essentially, reduce the effect of regularisation.
* Optimise speed.
* Simplify code - by default represent skills column-by-column as opposed to row by row.

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

The Hessian of the multivariate normal (w.r.t. means) is simply $\Sigma^{-1}$ ([source](https://stats.stackexchange.com/questions/27436/how-to-take-derivative-of-multivariate-normal-density)).

In terms of the likelihood with regards to match outcomes, the Jacobian and the Hessian for logistic regression can be found [here](https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function).

With the gradient and the Hessian, the Newton-Raphson method can be used for finding the MAP. 
