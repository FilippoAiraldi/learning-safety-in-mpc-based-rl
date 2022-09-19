import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import sklearn.gaussian_process as gp
from mpc.generic_mpc import subsevalf
from typing import Union


# from scipy.stats import norm
# 1.96 = norm.ppf((0.95 + 1) / 2) # because of abs value, but we need the tail


def cs_kernel_const(
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


def cs_kernel_rbf(
    length_scale: float,
    X: Union[np.ndarray, cs.SX, cs.MX, cs.DM],
    Y: Union[np.ndarray, cs.SX, cs.MX, cs.DM] = None,
    diag: bool = False
) -> Union[cs.SX, cs.MX, cs.DM, np.ndarray]:
    if not diag:
        if Y is None:
            Y = X
        n, m = X.shape[0], Y.shape[0]
        dists = cs.horzcat(
            *(cs.sum2((X - cs.repmat(Y[i, :].reshape((1, -1)), n, 1))**2)
              for i in range(m))
        )
        return np.exp(-0.5 * dists / (length_scale**2))
    else:
        assert Y is None
        return np.ones((X.shape[0], 1))


class GPR(cs.Callback):
    def __init__(self, name: str, gpr: gp.GaussianProcessRegressor, opts=None):
        if opts is None:
            opts = {}
        self.gpr = gpr
        cs.Callback.__init__(self)
        self.construct(name, opts)

    def eval(self, arg):
        return self.gpr.predict(np.array(arg[0]))


if __name__ == '__main__':
    # =========================================================== #
    # Multi-dimension GP (sklearn's GP) and reproducing in CasADi #
    # =========================================================== #

    # create data
    # X = np.stack((
    #     np.linspace(start=0, stop=10, num=1_000),
    #     np.linspace(start=-5, stop=5, num=1_000)), axis=-1)
    X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
    y = (X * np.sin(X)).sum(axis=-1)

    # create training data
    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(y.size), size=60, replace=False)
    X_train, y_train = X[training_indices], y[training_indices]
    noise_std = 0.75
    y_train_noisy = y_train + rng.normal(
        loc=0.0, scale=noise_std, size=y_train.shape)

    # create kernel and GP and fit it
    kernel = 2 * gp.kernels.RBF(length_scale=1.0,
                                length_scale_bounds=(1e-2, 1e2))
    gpr = gp.GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=9,  # alpha=noise_std**2,
    )
    gpr.fit(X_train, y_train_noisy)

    # perform numerical prediction
    mean_prediction, std_prediction = gpr.predict(X, return_std=True)

    plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
    plt.errorbar(
        X_train,
        y_train_noisy,
        noise_std,
        linestyle="None",
        color="tab:blue",
        marker=".",
        markersize=10,
        label="Observations",
    )
    plt.plot(X, mean_prediction, label="Mean prediction")
    plt.fill_between(
        X.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        color="tab:orange",
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.title("Gaussian process regression on a noisy dataset")
    plt.show()

    # perform numerical prediction manually
    k = gpr.kernel_(gpr.X_train_, X)
    K = gpr.kernel_(gpr.X_train_) + gpr.alpha * np.eye(gpr.X_train_.shape[0])
    K_inv = np.linalg.inv(K)
    y_mean1 = 0 + k.T @ K_inv @ (gpr.y_train_ - 0)
    y_std1 = np.sqrt(gpr.kernel_.diag(X) - np.diag(k.T @ K_inv @ k))

    # perform symbolical prediction (changes if the kernel is modified)
    cs_kernel = lambda X, Y = None: \
        cs_kernel_const(gpr.kernel_.k1.constant_value, X, Y) * \
        cs_kernel_rbf(gpr.kernel_.k2.length_scale, X, Y)
    cs_kernel_diag = lambda X: \
        cs_kernel_const(gpr.kernel_.k1.constant_value, X, diag=True) * \
        cs_kernel_rbf(gpr.kernel_.k2.length_scale, X, diag=True)
    Xsym = cs.SX.sym('X', *X.shape)
    k = cs_kernel(gpr.X_train_, Xsym)
    y_mean_sym = k.T @ gpr.alpha_
    y_std_sym = cs.sqrt(
        cs_kernel_diag(Xsym) -
        cs.vertcat(*(k[:, i].T @ K_inv @ k[:, i] for i in range(k.shape[1])))
    )

    quit(0)

    # ==================================================== #
    # CasADi Callbacks for external library (sklearn's GP) #
    # ==================================================== #

    # Create data points: a noisy sine wave
    N = 20
    np.random.seed(0)
    data = np.linspace(0, 10, N).reshape((N, 1))
    value = np.sin(data) + np.random.normal(0, 0.1, (N, 1))

    # use sklearn
    gpr = gp.GaussianProcessRegressor(
        kernel=(
            gp.kernels.ConstantKernel() + gp.kernels.DotProduct() +
            gp.kernels.WhiteKernel() + gp.kernels.RBF()),
        n_restarts_optimizer=9
    )
    gpr.fit(data, value)

    # Sample the resulting regression finely for plotting
    xf = np.linspace(0, 10, 10 * N).reshape((-1, 1))
    mean, sigma = gpr.predict(xf, return_std=True)

    # Plotting
    plt.fill_between(xf.squeeze(), mean - 3 * sigma, mean + 3 * sigma,
                     color='#aaaaff', label='fit 3 sigma bounds')
    plt.plot(data, value, 'ro', label='data')
    plt.plot(xf, mean, 'k-', label='fit')
    plt.xlabel('independant variable')
    plt.ylabel('dependant variable')
    plt.legend()

    # Package the resulting regression model in a CasADi callback

    # Instantiate the Callback (make sure to keep a reference to it!)
    gpr = GPR('GPR', {'enable_fd': True})
    print(gpr)

    # Find the minimum of the regression model
    x = cs.MX.sym('x')
    solver = cs.nlpsol('solver', 'ipopt', {'x': x, 'f': x, 'g': 0.75 - gpr(x)})
    res = solver(x0=3, lbg=-np.inf, ubg=0)

    plt.plot(float(res['x']), float(gpr(res['x'])), 'k*', markersize=10,
             label='Function minimum by CasADi/Ipopt')
    plt.legend()
    plt.show()

    quit(0)
