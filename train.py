import envs
import agents
import util
from tqdm import tqdm


# TODO:
1. record at each time step the quantities above, or record the solution object, in an internal temporary buffer of the agent
2. at the end of an episode, consolidate these quantities inside the replaymem (in order to maintain the separation between episodes!)
- perform solution.value and lstsq to find dpidtheta
- save everthing to replay memory
3. then call update


if __name__ == '__main__':
    # TODO: argparse
    # - fix seed
    # - training episodes
    util.set_np_mpl_defaults()

    # initialize env and agent
    max_episode_steps = 50
    env = envs.QuadRotorEnv.get_wrapped(
        max_episode_steps=max_episode_steps)

    # agent = agents.QuadRotorPIAgent(env=env, agentname='PI')
    agent = agents.QuadRotorDPGAgent(env=env, agentname='DPG')

    # simulate
    env.reset(seed=4)
    for _ in tqdm(range(max_episode_steps), total=max_episode_steps):
        # compute V(s)
        u_opt, _, solution = agent.predict(deterministic=False)
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
