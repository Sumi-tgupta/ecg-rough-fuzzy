import numpy as np
import pandas as pd

from modules.rough_set import Discretizer, RoughSetSelector, quick_reduct


def test_discretizer_fits_train_cuts_and_imputes_nonfinite_values():
    train = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, np.nan], "b": [1, 1, 2, 2, 3]})
    disc = Discretizer(n_bins=3).fit(train)
    transformed = disc.transform(pd.DataFrame({"b": [2, np.inf], "a": [2.5, np.nan]}))
    assert transformed.shape == (2, 2)
    assert np.issubdtype(transformed.dtype, np.integer)


def test_quick_reduct_returns_nonempty_reduct_even_when_dependency_is_zero():
    D = np.zeros((12, 2), dtype=int)
    y = np.array(["N", "V"] * 6)
    result = quick_reduct(D, y, n_labels=2)
    assert len(result["reduct"]) == 1
    assert result["gamma_reduct"] == 0.0


def test_rough_set_selector_reports_reduct_and_approximations():
    X = pd.DataFrame({"a": [0, 0, 1, 1, 2, 2], "b": [0, 1, 0, 1, 0, 1]})
    y = np.array(["N", "N", "S", "S", "V", "V"])
    selector = RoughSetSelector(n_bins=3, max_per_class=20, seed=3).fit(X, y)
    assert selector.selected_
    assert set(selector.selected_).issubset(set(X.columns))
    assert 0.0 <= selector.gamma_reduct_ <= 1.0
    assert len(selector.approximations_) == len(set(y))
