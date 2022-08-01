import envs
import agents
import util
from tqdm import tqdm


if __name__ == '__main__':
    # TODO: argparse
    # - fix seed
    # - training episodes
    # vectorize environment and solve MPC in parallel
    util.set_np_mpl_defaults()

    # initialize env and agent
    episodes = 2
    max_episode_steps = 50
    env: envs.QuadRotorEnv = envs.QuadRotorEnv.get_wrapped(
        max_episode_steps=max_episode_steps)

    # agent = agents.QuadRotorPIAgent(env=env, agentname='PI', seed=69)
    agent = agents.QuadRotorDPGAgent(env=env, agentname='DPG', seed=69)

    # simulate
    for i in range(episodes):
        # reset env
        env.reset(seed=42 * i)

        # simulate this episode
        for _ in tqdm(range(max_episode_steps), total=max_episode_steps):
            # compute V(s)
            u, _, solution = agent.predict(deterministic=False,
                                           perturb_gradient=False)
            assert solution.success

            # step environment
            _, r, done, info = env.step(u)

            # save transition
            agent.save_transition((env.x, u, r), solution)

            # check if episode is done
            if done:
                break

        # perform RL update
        agent.consolidate_episode_experience()

        # reduce exploration strength
        agent.perturbation_strength *= 0.97

    # save results and launch plotting (is blocking)
    fn = util.save_results(env=env)
    import os
    os.system(f'python visualization.py -fn {fn}')
