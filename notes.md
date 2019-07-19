# Notes

## 2019-07-17

In commit `bff2c81f43eaafe8747fcbfeaed5d64d2b97bef4`, the Gaussian process works and is able to perform Metropolis-Hastings sampling on a small dataset with increasing likelihood. See [this notebook](notebooks/02-test_gp_hyperparams.20190718.ipynb).

However, due to the Gaussian process constraints and the information-poor nature of the per-game outcomes, the sampling is completely dominated by the Gaussian prior and not at all by the match outcomes.

An improved approach might be to first find the MLE of the model, the sample around it using Metropolis-Hastings.

An initial attempt to brute force the fit using BFGS failed. On a small TI9 qualifers dataset, BFGS fitting did not complete after more than 10 hours.

Separately, line profiling was performed on [a script](src/scripts/fit_ti9.py) that fitted the GP on the same TI9 dataset. The results show that computing the multivariate Gaussian log-probability density function consumes the majority of the time during fitting. Since this function is implemented in `scipy.stats.multivariate_normal`, it is unclear how this function could be sped up.
