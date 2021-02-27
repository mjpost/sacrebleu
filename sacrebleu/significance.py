import os
from typing import List, Tuple

import numpy as np

# Set numpy RNG's seed for significance testing
# If not given -> Fix to 12345
# If given but <= 0, don't fix the seed i.e. leave it uninitialized
SACREBLEU_SEED = int(os.environ.get('SACREBLEU_SEED', '12345'))
if SACREBLEU_SEED > 1:
    np.random.seed(SACREBLEU_SEED)


class Bootstrap:
    @staticmethod
    def get_seed():
        return SACREBLEU_SEED

    @staticmethod
    def resample_stats(stats: np.ndarray, n_bootstrap: int = 1000):
        idxs = np.random.choice(
            len(stats), size=(n_bootstrap, len(stats)), replace=True)
        return stats[idxs]

    @staticmethod
    def confidence_interval(estimates: List[float]) -> Tuple[float, float]:
        """Computes mean and the 95% confidence interval from resampled estimates.

        :param estimates: A list of floats computed through bootstrap resampling.
        :return: A tuple of (mean, CI).
        """
        estimates = np.array(estimates)
        mean, stdev = np.mean(estimates), np.std(estimates)
        ci = (1.96 * stdev) / (len(estimates) ** 0.5)
        return mean, ci
