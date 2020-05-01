import numpy as np
import scipy.stats as stats


"""Helper functions."""

def lse(arr):
    maxval = arr.max()
    return maxval + np.log(np.sum(np.exp(arr - maxval)))


def lse_rows(mat):
    maxval = mat.max(axis=1)
    return maxval + np.log(np.exp(mat - maxval[:,np.newaxis]).sum(axis=1))


def lse_cols(mat):
    maxval = mat.max(axis=0)
    return maxval + np.log(np.exp(mat - maxval[np.newaxis,...]).sum(axis=0))


def gen_prob_vecs(nrow, ncol, alpha_min=1, alpha_max=1):    
    alphas = stats.uniform(alpha_min, alpha_max - alpha_min).rvs(nrow)
    return np.array([stats.dirichlet([alpha] * ncol).rvs()[0] for alpha in alphas])  