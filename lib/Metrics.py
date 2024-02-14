import numpy as np
import pandas as pd
from scipy.stats import norm

def nll(true, mean=None, std=None, bins=False):
        if isinstance(true, pd.DataFrame):
            mean = true['Pred']
            std = true['Std']
            true = true['True']

        # Calculate NLL
        nll_values = norm.logpdf(true, loc=mean, scale=std)
        return -np.mean(nll_values)

def mae(true, mean=None, std=None, bins=False):
        if isinstance(true, pd.DataFrame):
            mean = true['Pred']
            std = true['Std']
            true = true['True']

        # Calculate NLL
        mae = np.mean(np.abs(true-mean))
        return mae

def mb_log(true, mean=None, std=None, bins=False):
    if bins:
        correct_bin = np.floor(true['True']*10)/10
        correct_bin = pd.DataFrame(index=correct_bin.index, data=[float("{:.1f}".format(v)) for v in correct_bin.values])

        cols = [float("{:.1f}".format(v)) for v in true.columns[:-1]]
        cols.append('True')
        true.columns = cols

        mbl = np.asarray([])
        for idx in true.index:
            bin_val = correct_bin.loc[idx][0]
            lower = float("{:.1f}".format(bin_val - 0.5))
            upper = float("{:.1f}".format(bin_val + 0.5))
            mbl = np.append(mbl, np.log(true.loc[idx, lower:upper].sum()))

        return mbl
    try:
        if isinstance(true, pd.DataFrame):
            mean = true['Pred']
            std = true['Std']
            true = true['True']

        dist = norm(loc=mean, scale=std)

        mbl = np.log((dist.cdf(true + 0.6) - dist.cdf(true - 0.5)))
        mbl[np.invert(np.isfinite(mbl))] = -10
        mbl[mbl < -10] = -10

        return mbl

    except Exception as e:
        print(e)
        return -10

def skill(true, mean=None, std=None, bins=False):
    """
    Calculate the skill score based on the mean logarithm of the likelihood.

    Parameters:
    - true: Ground truth values.
    - mean: Mean predictions.
    - std: Standard deviations of predictions.
    - bins: If True, perform bin-based skill score calculation.

    Returns:
    - Skill score.
    """
    return np.exp(mb_log(true, mean, std, bins).mean())