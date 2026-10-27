"""The four classifiers with scikit-learn default hyperparameters (paper Section 2.4)."""

import time
import warnings

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

MODEL_NAMES = ["LR", "LDA", "SVM", "MLP"]


def make_model(name: str, seed: int, max_iter: int | None = None, device: str = "cpu"):
    """Defaults everywhere; only random_state (and LR max_iter for the alt run) is set."""
    if device == "cuda" and name in ("LR", "SVM"):
        return _make_cuml(name, max_iter)
    if name == "LR":
        return LogisticRegression(random_state=seed, **({"max_iter": max_iter} if max_iter else {}))
    if name == "LDA":
        return LinearDiscriminantAnalysis()
    if name == "SVM":
        return SVC(probability=True, random_state=seed)   # probability only for the ROC/AUC
    if name == "MLP":
        return MLPClassifier(random_state=seed)
    raise ValueError(f"unknown model {name!r}")


def _make_cuml(name: str, max_iter: int | None):
    """cuML equivalents. Their defaults are close to, not identical to, scikit-learn's
    (cuML LR uses the QN solver; SVC Platt scaling differs), so GPU numbers are flagged."""
    if name == "LR":
        from cuml.linear_model import LogisticRegression as CuLR
        return CuLR(C=1.0, **({"max_iter": max_iter} if max_iter else {"max_iter": 100}))
    from cuml.svm import SVC as CuSVC
    return CuSVC(C=1.0, kernel="rbf", gamma="scale", probability=True)


def fit_predict(model, X_train, y_train, X_test) -> dict:
    """Fit, predict and score; records whether a ConvergenceWarning was raised."""
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    pred = np.asarray(_host(model.predict(X_test))).astype(np.int64)
    proba = np.asarray(_host(model.predict_proba(X_test)), dtype=np.float64)
    # Models trained without a class (e.g. a fold with no Q) return fewer columns.
    classes = np.asarray(_host(model.classes_)).astype(np.int64)
    full = np.zeros((len(X_test), 5))
    full[:, classes] = proba
    return {
        "pred": pred, "proba": full, "fit_seconds": fit_s,
        "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
        "n_iter": int(np.max(_host(getattr(model, "n_iter_", -1)))),
    }


def _host(a):
    return a.get() if hasattr(a, "get") else a
