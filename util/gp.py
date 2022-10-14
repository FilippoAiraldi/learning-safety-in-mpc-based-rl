import math
from typing import Callable, Optional, Union
import casadi as cs
import numpy as np
from joblib import Parallel
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import check_is_fitted


KERNEL_PARAMS_DICT = {
    kernels.RBF: ('length_scale',),
    kernels.Matern: ('nu', 'length_scale',),
    kernels.ConstantKernel: ('constant_value',),
    kernels.WhiteKernel: ('noise_level',)
}


class CasadiKernels:
    '''
    Static class implementing in CasADi the GP kernels avaiable in sklearn.
    Each kernel type retains the same nomenclature.
    '''

    @staticmethod
    def ConstantKernel(
        pars: tuple[float],
        X: Union[np.ndarray, cs.SX, cs.MX, cs.DM],
        Y: Union[np.ndarray, cs.SX, cs.MX, cs.DM] = None,
        diag: bool = False
    ) -> np.ndarray:
        val, = pars
        if not diag:
            if Y is None:
                Y = X
            return np.full((X.shape[0], Y.shape[0]), val)
        else:
            assert Y is None
            return np.full((X.shape[0], 1), val)

    @staticmethod
    def RBF(
        pars: tuple[Union[float, np.ndarray]],
        X: Union[np.ndarray, cs.SX, cs.MX, cs.DM],
        Y: Union[np.ndarray, cs.SX, cs.MX, cs.DM] = None,
        diag: bool = False
    ) -> Union[np.ndarray, cs.SX, cs.MX, cs.DM]:
        length_scale, = pars
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
    def Matern(
        pars: tuple[float, Union[float, np.ndarray]],
        X: Union[np.ndarray, cs.SX, cs.MX, cs.DM],
        Y: Union[np.ndarray, cs.SX, cs.MX, cs.DM] = None,
        diag: bool = False
    ) -> Union[np.ndarray, cs.SX, cs.MX, cs.DM]:
        nu, length_scale = pars
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
            dists = cs.horzcat(*(cs.sqrt(cs.sum2(
                    (X - cs.repmat(Y[i, :].reshape((1, -1)), n, 1))**2))
                for i in range(m)
            ))

            if nu == 0.5:
                K = np.exp(-dists)
            elif nu == 1.5:
                K = dists * math.sqrt(3)
                K = (1.0 + K) * np.exp(-K)
            elif nu == 2.5:
                K = dists * math.sqrt(5)
                K = (1.0 + K + K**2 / 3.0) * np.exp(-K)
            elif nu == np.inf:
                K = np.exp(-(dists**2) / 2.0)
            else:  # general case; expensive to evaluate
                raise ValueError('Invalud nu.')

            return K
        else:
            assert Y is None
            return np.ones((X.shape[0], 1))

    @staticmethod
    def WhiteKernel(
        pars: tuple[float],
        X: Union[np.ndarray, cs.SX, cs.MX, cs.DM],
        Y: Union[np.ndarray, cs.SX, cs.MX, cs.DM] = None,
        diag: bool = False
    ) -> np.ndarray:
        noise_level, = pars
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
        def _recursive(_krl):
            if isinstance(_krl, kernels.KernelOperator):
                k1 = _recursive(_krl.k1)
                k2 = _recursive(_krl.k2)
                if isinstance(_krl, kernels.Sum):
                    return lambda X, Y, diag: k1(X, Y, diag) + k2(X, Y, diag)
                if isinstance(_krl, kernels.Product):
                    return lambda X, Y, diag: k1(X, Y, diag) * k2(X, Y, diag)
                raise TypeError('Unrecognized kernel operator.')

            func = getattr(CasadiKernels, type(_krl).__name__)
            p = [getattr(_krl, a) for a in KERNEL_PARAMS_DICT[type(_krl)]]
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

    def predict(
        self, 
        X: np.ndarray, 
        return_std: bool = False, 
        return_cov: bool = False
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
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
