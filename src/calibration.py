"""SuddWatch - probability calibration and prior-shift correction (shared by training and the API)."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Platt:
    """Sigmoid calibration on the logit of the raw score, fitted on out-of-sample predictions only."""

    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p), y)
        self.base_prior = float(np.mean(y))
        return self

    def __call__(self, p):
        return self.lr.predict_proba(_logit(p))[:, 1]


def prior_shift(p, base, new):
    """Bayes prior correction: probabilities calibrated at prevalence `base`, re-weighted to prevalence `new`.

    `new` is the national positive rate of labels already known at the issue date, so the decision
    threshold adapts to the recent flood regime without using any future information.
    """
    p = np.asarray(p, dtype=float)
    r1, r0 = new / base, (1 - new) / (1 - base)
    return p * r1 / (p * r1 + (1 - p) * r0)
