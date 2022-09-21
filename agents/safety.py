import casadi as cs
import numpy as np
from joblib import Parallel
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import check_is_fitted
from typing import Any


# useful links:
# https://scikit-learn.org/stable/modules/gaussian_process.html
# http://gaussianprocess.org/gpml/chapters/RW.pdf
# https://web.casadi.org/blog/tensorflow/
# https://groups.google.com/g/casadi-users/c/gLJNzajFM6w


class GaussianProcessRegressorConstraintCallback(cs.Callback):
    def __init__(
        self,
        name: str,
        gpr: GaussianProcessRegressor,
        beta: float,
        center: float = 0.0,
        opts: dict[str, Any] = None
    ) -> None:
        if opts is None:
            opts = {}
        self._gpr = gpr
        self.beta = beta
        self._C = center
        cs.Callback.__init__(self)
        self.construct(name, opts)

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        assert 0.0 <= value <= 1.0, 'beta must be in range [0, 1].'
        self._beta = value
        self._beta_ppf = norm.ppf(value)

    def eval(self, arg: Any) -> Any:
        mean, std = self._gpr.predict(np.array(arg[0]), return_std=True)
        return (mean - self._C) + self._beta_ppf * std


class MultitGaussianProcessRegressor(MultiOutputRegressor):
    '''Custom multioutput regressor adapted to GP regression.'''

    def __init__(
        self,
        kernel: kernels.Kernel = None,
        *,
        alpha: float = 1e-10,
        optimizer: str = 'fmin_l_bfgs_b',
        n_restarts_optimizer: int = 0,
        normalize_y: bool = False,
        copy_X_train: bool = True,
        random_state: int = None,
        n_jobs: int = None
    ) -> None:
        estimator = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train,
            random_state=random_state
        )
        super().__init__(estimator, n_jobs=n_jobs)

    def predict(self, X, return_std=False, return_cov=False):
        '''See `GaussianProcessRegressor.predict`.'''
        check_is_fitted(self)
        y = Parallel(n_jobs=self.n_jobs)(
            delayed(e.predict)(
                X, return_std, return_cov
            ) for e in self.estimators_
        )
        if return_std or return_cov:
            return tuple(np.array(o).T for o in zip(*y))
        return np.asarray(y).T


def constraint_violation(
    *arrays_and_bounds: tuple[np.ndarray, np.ndarray]
) -> list[tuple[np.ndarray, np.ndarray]]:
    '''Computes constraint violations.

    Parameters
    ----------
    arrays_and_bounds : *tuple[array_like, array_like]
        Any number of `(array, bounds)` for which to compute the constraint 
        violations.
        For each tuple, `bounds` is an array of shape `(N, 2)`, where `N` is 
        the number of features, and the first and second columns are lower and
        upper bounds, respectively. `array` is an array of shape `(N, ...)`.
        A violation is defined as `x <= bound`.

    Returns
    -------
    cv : list[tuple[array_like, array_like]]
        For each tuple provided, returns a tuple of lower and upper bound 
        violations. 
    '''
    cv = []
    for a, bnd in arrays_and_bounds:
        a = np.asarray(a)
        lb = np.expand_dims(bnd[:, 0], tuple(range(1, a.ndim)))
        ub = np.expand_dims(bnd[:, 1], tuple(range(1, a.ndim)))
        cv.append((lb - a, a - ub))
    return cv
