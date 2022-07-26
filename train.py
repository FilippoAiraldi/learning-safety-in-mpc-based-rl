import numpy as np
import envs
import agents
import util
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=False)

    # initialize env and agent
    max_episode_steps = 50
    env: envs.QuadRotorEnv = envs.QuadRotorEnv.get_wrapped(
        max_episode_steps=max_episode_steps)
    agent = agents.QuadRotorPIAgent(env=env, agentname='PI')
    pars = {'perturbation': 0}

    # simulate
    env.reset()
    for _ in tqdm(range(max_episode_steps), total=max_episode_steps):
        # compute V(s)
        u_opt, solution = agent.solve_mpc('V', other_pars=pars)
        assert solution.success
        
        # step environment
        _, r, done, info = env.step(u_opt)

        # check if episode is done
        if done:
            break

    # plot
    util.plot.plot_trajectory_3d(env, 0)
    util.plot.plot_trajectory_in_time(env, 0)
    plt.show()
