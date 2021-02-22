from typing import List, Tuple

import numpy as np


def bootstrap_ci(estimates: List[float]) -> Tuple[float, float]:
    """Computes mean and the 95% confidence interval from resampled estimates.

    :param estimates: A list of floats computed through bootstrap resampling.
    :return: A tuple of (mean, CI).
    """
    estimates = np.array(estimates)
    mean, stdev = estimates.mean(), estimates.std()
    ci = (1.96 * stdev) / (len(estimates) ** 0.5)
    return mean, ci
