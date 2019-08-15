# Issues

## Open

1. Write classes for exploring data samples easily.
1. Further speed optimisation using `np.einsum()`.

## Completed

1. Optimise Python model fitting for speed.
    1. Simplify code - represent skills vector column-wise as opposed to row-wise. Use a list of numpy arrays for each player.
    2. Sample from the standard normal distribution and multiply it with a precomputed Cholesky-decomposition of the covariance matrix to yield the skills vectors.
    3. The prior probability vector can be fully computed in the standard normal space. The only time we need to transform the standard normal space vectors into "skill" vectors is when we need to compute match probabilities.
    4. Should write a class to handle Gaussian process prior probability calculation. cls.vec returns the _transformed_ data. cls.loglik returns the current log-likelihood based on _the standard multivariate normal distribution_.
1. Try a larger scaling factor for the logistic model (i.e. reduce the effect of the multivariate normal prior).
1. Write a class for storing samples and sets of samples. Refactor code to use this class.
1. Write a wrapper class for a matches table that performs sanity checks like column names.

## Abandoned

1. Convert the GP model into Stan.
