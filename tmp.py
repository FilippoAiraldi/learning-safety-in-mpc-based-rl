# import numpy as np
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF

# # create data
# X = np.stack((
#     np.linspace(start=0, stop=10, num=1_000),
#     np.linspace(start=-5, stop=5, num=1_000)), axis=-1)
# y = (X * np.sin(X)).sum(axis=-1)

# # create training data
# rng = np.random.RandomState(1)
# training_indices = rng.choice(np.arange(y.size), size=10, replace=False)
# X_train, y_train = X[training_indices], y[training_indices]
# noise_std = 0.75
# y_train_noisy = y_train + rng.normal(
#     loc=0.0, scale=noise_std, size=y_train.shape)

# # create kernel and GP and fit it
# kernel = 2 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
# gaussian_process = GaussianProcessRegressor(
#     kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9
# )
# gaussian_process.fit(X_train, y_train_noisy)

# # # perform numerical prediction
# # mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)


# # perform symbolical prediction (changes if the kernel is modified)
# import casadi as cs
# from mpc.generic_mpc import subsevalf
# from scipy.linalg import solve_triangular
# gp = gaussian_process

# #
# K_trans0 = gp.kernel_(X, gp.X_train_)
# y_mean0 = gp._y_train_std * (K_trans0 @ gp.alpha_) + gp._y_train_mean
# V0 = solve_triangular(gp.L_, K_trans0.T, lower=True)

# #
# Xsym = cs.SX.sym('X', *X.shape)  # X has fixed shape
# Ysym = cs.SX.sym('Y', *gp.X_train_.shape)  # X train has variable shape
# k1_value = cs.SX.sym('k1_c', 1, 1)
# k2_length_scale = cs.SX.sym('k1_length_scale', 1, 1)
# alpha = cs.SX.sym('alpha', gp.X_train_.shape[0], 1)  # alpha has variable shape
# y_train_std = cs.SX.sym('y_train_std', 1, 1)
# y_train_mean = cs.SX.sym('y_train_mean', 1, 1)
# L = cs.tril(cs.SX.sym('L', gp.X_train_.shape[0], gp.X_train_.shape[0]))

# k1 = cs.repmat(k1_value, Xsym.shape[0], Ysym.shape[0])
# dists = cs.horzcat(
#     *(cs.sum2((Xsym - cs.repmat(Ysym[i, :], Xsym.shape[0], 1))**2)
#       for i in range(Ysym.shape[0]))
# )
# k2 = cs.exp(-0.5 * dists / k2_length_scale**2)
# K_trans = k1 * k2
# y_mean = y_train_std * (K_trans @ alpha) + y_train_mean

# y_mean_ = subsevalf(
#     y_mean,
#     [Xsym, Ysym, k1_value, k2_length_scale, alpha, y_train_std, y_train_mean],
#     [X, gp.X_train_, gp.kernel_.k1.constant_value, gp.kernel_.k2.length_scale,
#      gp.alpha_, gp._y_train_std, gp._y_train_mean])

# # K_trans2_ = subsevalf(K_trans2, (Xsym, Ysym), (X, gp.X_train_))

# quit(0)
# ##############################################################################


import casadi as cs
import numpy as np
import matplotlib.pyplot as plt

# Create data points: a noisy sine wave
N = 20
np.random.seed(0)
data = np.linspace(0, 10, N).reshape((N, 1))
value = np.sin(data) + np.random.normal(0, 0.1, (N, 1))


# use sklearn
import sklearn.gaussian_process as gp
gaussian_process = gp.GaussianProcessRegressor(
    kernel=(
        gp.kernels.ConstantKernel() + gp.kernels.DotProduct() +
        gp.kernels.WhiteKernel() + gp.kernels.RBF()),
    n_restarts_optimizer=9
)
gaussian_process.fit(data, value)

# Sample the resulting regression finely for plotting
xf = np.linspace(0, 10, 10 * N).reshape((-1, 1))
mean, sigma = gaussian_process.predict(xf, return_std=True)


# Plotting
plt.fill_between(xf.squeeze(), mean - 3 * sigma, mean + 3 * sigma,
                 color='#aaaaff', label='fit 3 sigma bounds')
plt.plot(data, value, 'ro', label='data')
plt.plot(xf, mean, 'k-', label='fit')
plt.xlabel('independant variable')
plt.ylabel('dependant variable')
plt.legend()

# Package the resulting regression model in a CasADi callback


class GPR(cs.Callback):
    def __init__(self, name, opts=None):
        if opts is None:
            opts = {}
        cs.Callback.__init__(self)
        self.construct(name, opts)

    def eval(self, arg):
        return gaussian_process.predict(np.array(arg[0]))


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
