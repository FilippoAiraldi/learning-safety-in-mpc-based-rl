import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from mpc.generic_mpc import subsevalf
from scipy.linalg.lapack import dtrtri
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.model_selection import train_test_split
from util.gp import MultitGaussianProcessRegressor, \
    MultiGaussianProcessRegressorCallback, CasadiKernels

# from scipy.stats import norm
# 1.96 = norm.ppf((0.95 + 1) / 2) # because of abs value, but we need the tail


def reproducing_gp_in_casadi():
    # create data
    X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
    y = np.squeeze(X * np.sin(X))

    # create training data
    rng = np.random.RandomState(np.random.randint(0, 1000))
    training_indices = rng.choice(np.arange(y.shape[0]), size=6, replace=False)
    X_train, y_train = X[training_indices], y[training_indices]
    noise_std = 0.75
    y_train_noisy = y_train + rng.normal(
        loc=0.0, scale=noise_std, size=y_train.shape)

    # create kernel and GP and fit it
    kernel = 1 * \
        kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + \
        kernels.WhiteKernel()
    gpr = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=9, alpha=noise_std**2,
    )
    gpr.fit(X_train, y_train_noisy)

    # perform numerical prediction
    y_mean0, y_std0 = gpr.predict(X, return_std=True)

    # perform numerical prediction manually
    k = gpr.kernel_(gpr.X_train_, X)
    L_inv = dtrtri(gpr.L_, lower=True)[0]
    K_inv = L_inv.T @ L_inv
    y_mean1 = 0 + k.T @ K_inv @ (gpr.y_train_ - 0)
    y_std1 = np.sqrt(gpr.kernel_.diag(X) - np.diag(k.T @ K_inv @ k))

    # perform symbolical prediction (changes if the kernel is modified)
    Xsym = cs.SX.sym('X', *X.shape)
    kernel_func = CasadiKernels.sklearn2func(gpr.kernel_)
    k = kernel_func(gpr.X_train_, Xsym)
    V = L_inv @ k
    y_mean_sym = k.T @ gpr.alpha_
    y_std_sym = cs.sqrt(kernel_func(Xsym, diag=True) - cs.sum1(V**2).T)
    y_mean2 = subsevalf(y_mean_sym, Xsym, X)
    y_std2 = subsevalf(y_std_sym, Xsym, X)

    print(*(np.allclose(*o) for o in [
        (y_mean0, y_mean1), (y_mean1, y_mean2),
        (y_std0, y_std1), (y_std1, y_std2)]))

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
    plt.plot(X, y_mean0, label="Mean prediction")
    plt.fill_between(
        X.ravel(),
        y_mean0 - 1.96 * y_std0,
        y_mean0 + 1.96 * y_std0,
        color="tab:orange",
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.title("Gaussian process regression on a noisy dataset")
    plt.show()


def gp_as_casadi_callback():
    # Create data points: a noisy sine wave
    N = 10
    np.random.seed(0)
    data = np.linspace(0, 10, N).reshape((N, 1))
    value = 0.2 * data + np.sin(data) + np.random.normal(0, 0.1, (N, 1))

    # use sklearn
    gpr = GaussianProcessRegressor(
        kernel=(
            kernels.ConstantKernel() + kernels.DotProduct() +
            kernels.WhiteKernel() + kernels.RBF()
        ),
        n_restarts_optimizer=9
    )
    gpr.fit(data, value)

    # Sample the resulting regression finely for plotting
    xf = np.linspace(0, 10, 10 * N).reshape((-1, 1))
    mean, sigma = gpr.predict(xf, return_std=True)
    beta = 4

    # Plotting
    plt.fill_between(xf.squeeze(), mean - 3 * sigma, mean + beta * sigma,
                     color='#aaaaff', label=f'fit {beta} sigma bounds')
    plt.plot(data, value, 'ro', label='data')
    plt.plot(xf, mean, 'k-', label='fit')
    plt.xlabel('independant variable')
    plt.ylabel('dependant variable')
    plt.legend()

    # Instantiate the Callback (make sure to keep a reference to it!)
    gprcb = MultiGaussianProcessRegressorCallback(
        gpr=gpr,
        n_theta=data.shape[1], n_features=value.shape[1],
        opts={'enable_fd': True})  # required
    print(gprcb)

    # Find the minimum of the regression model
    x = cs.MX.sym('x')
    mean, std = gprcb(x)
    f = mean + beta * std
    opts = {
        'expand': False,  # required (or just omit)
        'print_time': True,
        'ipopt': {
            'max_iter': 100,
            'sb': 'yes',
            # debug
            'print_level': 5,
            'print_user_options': 'no',
            'print_options_documentation': 'no'
        }}
    solver = cs.nlpsol('solver', 'ipopt', {'x': x, 'f': -f}, opts)
    res = solver(x0=7)

    f_opt = np.squeeze(cs.evalf(cs.substitute(f, x, res['x'])))
    plt.plot(np.squeeze(res['x']), f_opt, 'k*', markersize=10,
             label='Function minimum by CasADi/Ipopt')
    plt.legend()
    plt.show()


def multioutput_gp():
    # create data
    X = np.stack((
        np.linspace(start=0, stop=10, num=1000),
        np.linspace(start=-5, stop=5, num=1000)
    ), axis=-1)
    y = np.stack((
        (X * np.sin(X)).sum(axis=-1),
        (X * np.cos(X)).sum(axis=-1)
    ), axis=-1)
    rng = np.random.RandomState(1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=6, random_state=rng)
    noise_std = 0.75
    y_train_noisy = y_train + rng.normal(
        loc=0.0, scale=noise_std, size=y_train.shape)

    # fit a multioutput GPR
    kwargs = {
        'kernel': 1 * kernels.RBF(
            length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
        'n_restarts_optimizer': 9,
        'alpha': noise_std**2
    }
    mask = np.ones(y.shape[0], dtype=bool)
    gpr = GaussianProcessRegressor(**kwargs)
    gpr.fit(X_train, y_train_noisy)
    score = gpr.score(X_test, y_test)
    y_mean, y_std = gpr.predict(X_test[:3], return_std=True)

    # fit a MultiOutputRegressor for each output
    mogpr = MultitGaussianProcessRegressor(**kwargs)
    mogpr.fit(X_train, y_train_noisy)
    score_mo = mogpr.score(X_test, y_test)
    y_mean_mo, y_std_mo = mogpr.predict(X_test[:3], return_std=True)

    print(score, score_mo)


if __name__ == '__main__':
    reproducing_gp_in_casadi()
    # gp_as_casadi_callback()
    # multioutput_gp()
