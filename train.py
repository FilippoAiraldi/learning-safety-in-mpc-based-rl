import envs
import agents
import util
from tqdm import tqdm

# TODO
# 1) argparse
#   - fix seed
#   - number of  training episodes
# 2) vectorize environment and solve MPC in parallel
#   - vectorize all wrappers then
#   - vectorize also agent
# 3) agent monitor (saves weights and dJdtheta)
# 4) live plotter


if __name__ == '__main__':
    # set up
    util.set_np_mpl_defaults()
    run_name = util.get_run_name()
    logger = util.create_logger(run_name)

    # initialize env and agent
    episodes = 2
    max_episode_steps = 50
    env: envs.QuadRotorEnv = envs.QuadRotorEnv.get_wrapped(
        max_episode_steps=max_episode_steps)

    # agent = agents.QuadRotorPIAgent(env=env, agentname='PI', seed=69)
    agent: agents.QuadRotorDPGAgent = agents.wrappers.RecordLearningData(
        agents.QuadRotorDPGAgent(env=env, agentname='DPG', seed=69))

    # simulate
    for i in range(episodes):
        # reset env
        env.reset(seed=42 * i)

        # simulate this episode
        for _ in tqdm(range(max_episode_steps), total=max_episode_steps):
            # compute V(s)
            u, _, solution = agent.predict(deterministic=False,
                                           perturb_gradient=False)
            assert solution.success, 'Unexpected MPC failure.'

            # step environment
            _, r, done, info = env.step(u)

            # save transition
            agent.save_transition((env.x, u, r), solution)

            # check if episode is done
            if done:
                break

        # perform RL update
        agent.consolidate_episode_experience()
        agent.update()

        # reduce exploration strength
        agent.perturbation_strength *= 0.97

        # log episode outcomes
        logger.debug(f'J={env.cum_rewards[-1]:.3f} - '\
                     f'||dJdtheta||={agent.update_gradient_norm[-1]:.3f} - ' +
                     agent.weights.values2str())

    # save results and launch plotting (is blocking)
    fn = util.save_results(filename=run_name, env=env,
                           weights=agent.weights.values(as_dict=True))
    import os
    os.system(f'python visualization.py -fn {fn}')
