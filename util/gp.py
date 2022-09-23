import casadi as cs
import numpy as np
from joblib import Parallel
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import check_is_fitted
from typing import Any, Callable, Optional, Union


KERNEL_PARAMS_DICT = {
    kernels.RBF: 'length_scale',
    kernels.ConstantKernel: 'constant_value',
    kernels.WhiteKernel: 'noise_level'
}


class CasadiKernels:
    '''
    Static class implementing in CasADi the GP kernels avaiable in sklearn.
    Each kernel type retains the same nomenclature.
    '''

    @staticmethod
    def ConstantKernel(
        val: float,
        X: Union[np.ndarray, cs.SX, cs.MX, cs.DM],
        Y: Union[np.ndarray, cs.SX, cs.MX, cs.DM] = None,
        diag: bool = False
    ) -> np.ndarray:
        if not diag:
            if Y is None:
                Y = X
            return np.full((X.shape[0], Y.shape[0]), val)
        else:
            assert Y is None
            return np.full((X.shape[0], 1), val)

    @staticmethod
    def RBF(
        length_scale: Union[float, np.ndarray],
        X: Union[np.ndarray, cs.SX, cs.MX, cs.DM],
        Y: Union[np.ndarray, cs.SX, cs.MX, cs.DM] = None,
        diag: bool = False
    ) -> Union[np.ndarray, cs.SX, cs.MX, cs.DM]:
        if not diag:
            l = np.asarray(length_scale).reshape(1, -1)
            n = X.shape[0]
            X = X / np.tile(l, (n, 1))
            if Y is None:
                m = n
                Y = X
            else:
                m = Y.shape[0]
                Y = Y / np.tile(l, (m, 1))
            dists = cs.horzcat(*(
                cs.sum2((X - cs.repmat(Y[i, :].reshape((1, -1)), n, 1))**2)
                for i in range(m)
            ))
            return np.exp(-0.5 * dists)
        else:
            assert Y is None
            return np.ones((X.shape[0], 1))

    @staticmethod
    def WhiteKernel(
        noise_level: float,
        X: Union[np.ndarray, cs.SX, cs.MX, cs.DM],
        Y: Union[np.ndarray, cs.SX, cs.MX, cs.DM] = None,
        diag: bool = False
    ) -> np.ndarray:
        if not diag:
            return (
                (noise_level * np.eye(X.shape[0]))
                if Y is None else
                np.zeros((X.shape[0], Y.shape[0]))
            )
        assert Y is None
        return np.full((X.shape[0], 1), noise_level)

    @staticmethod
    def sklearn2func(
        kernel: Union[kernels.Kernel, kernels.KernelOperator]
    ) -> Callable[[
        Union[np.ndarray, cs.SX, cs.MX, cs.DM],
        Optional[Union[np.ndarray, cs.SX, cs.MX, cs.DM]],
        Optional[bool]
    ],
        Union[np.ndarray, cs.SX, cs.MX, cs.DM]
    ]:
        def _recursive(_kernel):
            if isinstance(_kernel, kernels.KernelOperator):
                k1 = _recursive(_kernel.k1)
                k2 = _recursive(_kernel.k2)
                if isinstance(_kernel, kernels.Sum):
                    return lambda X, Y, diag: k1(X, Y, diag) + k2(X, Y, diag)
                if isinstance(_kernel, kernels.Product):
                    return lambda X, Y, diag: k1(X, Y, diag) * k2(X, Y, diag)
                raise TypeError('Unrecognized kernel operator.')

            func = getattr(CasadiKernels, type(_kernel).__name__)
            p = getattr(_kernel, KERNEL_PARAMS_DICT[type(_kernel)])
            return lambda X, Y, diag: func(p, X, Y, diag)

        out = _recursive(kernel)
        return lambda X, Y=None, diag=False: out(X, Y, diag)


class MultitGaussianProcessRegressor(MultiOutputRegressor):
    '''
    Custom multi-regressor adapted to GP regression.

    Useful links:
    - https://scikit-learn.org/stable/modules/gaussian_process.html
    - http://gaussianprocess.org/gpml/chapters/RW.pdf
    '''

    estimators_: list[GaussianProcessRegressor]

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
