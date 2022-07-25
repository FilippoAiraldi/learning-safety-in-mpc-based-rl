import envs
import numpy as np
import util
import mpc


if __name__ == '__main__':
    # TODO: parse arguments

    #
    env = envs.QuadRotorEnv.get_wrapped(soft_state_con=True)
    # V = mpc.QuadRotorMPC(Np=20, type='V')

    env.reset(seed=42)
    done = False
    while not done:
        a = np.array([0, 0, 5])
        obs, r, done, _ = env.step(a)

    #
    util.plot.plot_trajectory(env, 0)
    

    print('DONE')
