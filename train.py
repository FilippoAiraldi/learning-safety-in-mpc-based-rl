import envs
import agents
import util
from tqdm import tqdm


if __name__ == '__main__':
    util.set_np_mpl_defaults()

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

    # save results and launch plotting (is blocking)
    fn = util.save_results(env=env)
    import os
    os.system(f'python visualization.py -fn {fn}')
