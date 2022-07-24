import envs
import numpy as np
import plot


if __name__ == '__main__':
    # TODO: parse arguments

    #
    env = envs.QuadRotorEnv.get_wrapped()
    env.reset(seed=42)
    done = False
    while not done:
        a = np.array([0, 0, 5])
        obs, r, done, _ = env.step(a)

    #
    plot.plot_trajectory(env, 0)
    plot.show()

    print('DONE')
