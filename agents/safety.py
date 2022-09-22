import casadi as cs
import numpy as np
from joblib import Parallel
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import check_is_fitted
from typing import Any


class MultitGaussianProcessRegressor(MultiOutputRegressor):
    '''Custom multi-regressor adapted to GP regression.'''

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


class MultiGaussianProcessRegressorCallback(cs.Callback):
    '''
    Custom callback to call the multi-GP regressor from CasADi.

    Useful links:
    - https://scikit-learn.org/stable/modules/gaussian_process.html
    - http://gaussianprocess.org/gpml/chapters/RW.pdf
    - https://web.casadi.org/blog/tensorflow/
    - https://groups.google.com/g/casadi-users/c/gLJNzajFM6w
    '''

    def __init__(
        self,
        gpr: MultitGaussianProcessRegressor,
        n_theta: int,
        n_features: int,
        opts: dict[str, Any] = None
    ) -> None:
        cs.Callback.__init__(self)
        self._gpr = gpr
        self._n_theta = n_theta
        self._n_features = n_features
        if opts is None:
            opts = {}
        self.construct('MGPRCB', opts)

    def get_n_in(self) -> int:
        return 1  # theta

    def get_n_out(self) -> int:
        return 2  # mean, std

    def get_name_in(self, i: int) -> str:
        return 'theta'

    def get_name_out(self, i: int) -> str:
        return 'mean' if i == 0 else 'std'

    def get_sparsity_in(self, i: int) -> cs.Sparsity:
        return cs.Sparsity.dense(self._n_theta, 1)

    def get_sparsity_out(self, i: int) -> cs.Sparsity:
        return cs.Sparsity.dense(self._n_features, 1)

    def eval(self, arg: Any) -> Any:
        theta = np.array(arg[0])
        if theta.shape[0] == self._n_theta:
            theta = theta.T
        mean, std = self._gpr.predict(theta, return_std=True)
        return mean.T, std.T


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
