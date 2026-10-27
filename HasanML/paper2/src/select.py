"""Min-max scaling and SelectKBest(f_classif) feature selection.

Note: this module is `src.select`; run the code as a package (`python -m src...`) so it
never shadows the standard-library `select` module.
"""

import warnings

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler


def fit_scaler(X_fit: np.ndarray) -> MinMaxScaler:
    return MinMaxScaler(feature_range=(0, 1)).fit(X_fit)


def fit_selector(X_fit: np.ndarray, y_fit: np.ndarray, k: int) -> SelectKBest:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # constant features -> NaN F-score
        return SelectKBest(f_classif, k=k).fit(X_fit, y_fit)


def fscore_table(selector: SelectKBest, names: list[str]) -> list[tuple[str, float, bool]]:
    """(feature, F-score, selected) sorted by F-score, highest first."""
    scores = np.nan_to_num(selector.scores_, nan=0.0)
    mask = selector.get_support()
    order = np.argsort(scores)[::-1]
    return [(names[i], float(scores[i]), bool(mask[i])) for i in order]
